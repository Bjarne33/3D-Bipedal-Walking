# Copyright 2026 Bjarne
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""rqt controls for isolated floating PD step experiments."""

import argparse
import math
from pathlib import Path
from queue import Empty
import shutil
import sys
import tempfile

from python_qt_binding.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from python_qt_binding.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)
from rqt_gui_py.plugin import Plugin

from humanoid_tuning.plots import write_layout
from humanoid_tuning.session import Session, Settings


def spinbox(lower, upper, value, decimals=4):
    """Create a numeric editor without hidden unit conversion.

    Args:
        lower: Minimum accepted value.
        upper: Maximum accepted value.
        value: Initial displayed value.
        decimals: Decimal precision.

    Returns:
        Configured spin box.
    """
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(lower, upper)
    box.setValue(value)
    return box


def position_range(box, limit):
    """Keep displayed degree bounds representable inside the exact URDF limits.

    Args:
        box: Position spin box.
        limit: Native-radian URDF joint limits.
    """
    scale = 10 ** box.decimals()
    box.setRange(math.ceil(math.degrees(limit.lower) * scale) / scale,
                 math.floor(math.degrees(limit.upper) * scale) / scale)


class TuningPanel(Plugin):
    """Configure gains, holding poses and step trials, with separate PlotJuggler plots."""

    def __init__(self, context):
        """Build the panel and start a private asynchronous ROS worker.

        Args:
            context: rqt plugin context and optional --simulator argument.
        """
        super().__init__(context)
        self.setObjectName('HumanoidTuning')
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--simulator', choices=['gazebo', 'mujoco'], default='gazebo')
        args, _ = parser.parse_known_args(context.argv())
        self.session = Session(args.simulator)
        self.applied = {}
        self.state = 'disconnected'
        self.auto_plots = True
        self.plot_restart = False
        self.closing = False
        self.directory = tempfile.TemporaryDirectory(prefix='humanoid-tuning-')
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.finished.connect(self._plots_finished)
        self.process.errorOccurred.connect(self._plots_error)
        self.widget = QWidget()
        self.widget.setWindowTitle(f'Humanoid PD tuning - {args.simulator}')
        layout = QVBoxLayout(self.widget)
        layout.addWidget(QLabel(
            f'{args.simulator}: pinned base, one joint per test. Gains are native SI values.'))
        self.status = QLabel('Connecting...')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        form = QFormLayout()
        self.joint = QComboBox()
        self.joint.addItems(self.session.names)
        form.addRow('Test joint', self.joint)
        self.p_gain = spinbox(0.0001, 1e9, 1.0)
        self.d_gain = spinbox(0, 1e9, 0)
        self.applied_label = QLabel('Not confirmed')
        form.addRow('P (N m/rad)', self.p_gain)
        form.addRow('D (N m s/rad)', self.d_gain)
        form.addRow('Confirmed applied P / D', self.applied_label)
        self.target = spinbox(-180, 180, 0)
        form.addRow('Absolute step target (deg)', self.target)
        self.timings = {}
        for key, label, value in (
                ('prepare', 'Prepare duration (s)', 2.0),
                ('window', 'Settled window (s)', 0.5),
                ('timeout', 'Prepare timeout (s)', 15.0),
                ('observe', 'Observation duration (s)', 5.0),
                ('position_tolerance', 'Settled position tolerance (deg)', math.degrees(0.02)),
                ('velocity_tolerance', 'Settled velocity tolerance (deg/s)', math.degrees(0.02))):
            box = spinbox(0.000001, 3600, value, 6)
            self.timings[key] = box
            form.addRow(label, box)
        layout.addLayout(form)
        self.table = QTableWidget(len(self.session.names), 3)
        self.table.setHorizontalHeaderLabels(['Joint', 'Holding position (deg)', 'Plot'])
        self.holding = {}
        self.checks = {}
        for row, name in enumerate(self.session.names):
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 0, item)
            limit = self.session.limits[name]
            box = spinbox(math.degrees(limit.lower), math.degrees(limit.upper), 0, 6)
            position_range(box, limit)
            box.valueChanged.connect(self._invalidate)
            self.holding[name] = box
            self.table.setCellWidget(row, 1, box)
            check = QCheckBox()
            check.setChecked(row == 0)
            check.toggled.connect(self._selection_changed)
            self.checks[name] = check
            self.table.setCellWidget(row, 2, check)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.buttons = {}
        for key, label, handler in (
                ('connect', 'Reconnect', lambda: self.session.submit('connect')),
                ('apply', 'Apply gains', self._apply),
                ('prepare', 'Prepare', self._prepare),
                ('start', 'Start step', self._start),
                ('stop', 'Stop (pause)', lambda: self.session.submit('stop')),
                ('plots', 'Open / reset plots', self._open_plots),
                ('export', 'Export gains...', self._export)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            self.buttons[key] = button
            buttons.addWidget(button)
        layout.addLayout(buttons)
        instructions = QLabel(
            'PlotJuggler 4.0: start ROS 2 Topic Subscriber, select the checked joints\' '
            '/humanoid_tuning/.../state topics and enable embedded/header timestamps. '
            'Then load the generated layout (use Copy layout path). '
            'Changing plot selection opens a fresh window to avoid mixing histories.')
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        copy_layout = QPushButton('Copy layout path')
        copy_layout.clicked.connect(self._copy_layout)
        layout.addWidget(copy_layout)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        layout.addWidget(self.log)
        context.add_widget(self.widget)
        self.joint.currentTextChanged.connect(self._joint_changed)
        self._joint_changed()
        self.p_gain.valueChanged.connect(self._gain_edited)
        self.d_gain.valueChanged.connect(self._gain_edited)
        self.timer = QTimer(self.widget)
        self.timer.timeout.connect(self._drain)
        self.timer.start(50)
        self.session.start()
        self._enable()

    def _settings(self):
        """Read the UI values and convert settling tolerances to native units.

        Returns:
            Experiment settings with ROS-radian tolerances.
        """
        values = {key: box.value() for key, box in self.timings.items()}
        for key in ('position_tolerance', 'velocity_tolerance'):
            values[key] = math.radians(values[key])
        return Settings(**values)

    def _invalidate(self):
        """Disable local readiness when the user edits the prepared holding pose."""
        if self.state == 'ready':
            self.state = 'idle'
            self.status.setText('Holding pose changed. Prepare again.')
            self._enable()

    def _gain_edited(self):
        """Mark unapplied edits without claiming that the controller has changed."""
        values = self.applied.get(self.joint.currentText())
        if values is not None:
            pending = values != (self.p_gain.value(), self.d_gain.value())
            suffix = ' (edited values NOT applied)' if pending else ''
            self.applied_label.setText(
                f'{values[0]:g} / {values[1]:g}{suffix}')

    def _joint_changed(self):
        """Update native gain fields, target bounds and required test-joint plot."""
        name = self.joint.currentText()
        limit = self.session.limits[name]
        position_range(self.target, limit)
        if name in self.applied:
            self.p_gain.setValue(self.applied[name][0])
            self.d_gain.setValue(self.applied[name][1])
        self._gain_edited()
        for joint, check in self.checks.items():
            check.setEnabled(joint != name)
        self.checks[name].setChecked(True)
        self._selection_changed()

    def _selection_changed(self):
        """Select stable joint-specific telemetry and discard old plot histories."""
        names = [name for name, check in self.checks.items() if check.isChecked()]
        self.session.submit('select', names)
        if self.process.state() != QProcess.NotRunning:
            self._open_plots()

    def _apply(self):
        """Submit a selected-joint atomic gain application."""
        self.session.submit(
            'apply', self.joint.currentText(), self.p_gain.value(), self.d_gain.value())
        self.state = 'applying'
        self._enable()

    def _prepare(self):
        """Submit a complete editable holding pose and settling settings."""
        pose = {name: math.radians(box.value()) for name, box in self.holding.items()}
        self.session.submit('prepare', pose, self._settings())
        self.state = 'preparing'
        self._enable()

    def _start(self):
        """Submit an absolute selected-joint target in ROS radians."""
        self.session.submit('start', self.joint.currentText(),
                            math.radians(self.target.value()), self._settings())
        self.state = 'running'
        self._enable()

    def _export(self):
        """Choose a new file for complete native configuration export."""
        filename, _ = QFileDialog.getSaveFileName(
            self.widget, 'Export applied backend gains to a NEW file', '', 'YAML (*.yaml)')
        if filename:
            self.session.submit('export', filename)

    def _drain(self):
        """Apply worker notifications on the Qt thread only."""
        while True:
            try:
                kind, value = self.session.events.get_nowait()
            except Empty:
                break
            if kind == 'status':
                self.state, text = value
                self.status.setText(f'{self.state}: {text}')
                self.log.append(text)
            elif kind == 'gains':
                self.applied = value
                self._joint_changed()
                if self.auto_plots:
                    self.auto_plots = False
                    QTimer.singleShot(500, self._open_plots)
            else:
                self.log.append(str(value))
        self._enable()

    def _enable(self):
        """Gate operations and editors during backend transitions and active tests."""
        busy = self.state in ('connecting', 'applying', 'preparing', 'running')
        connected = bool(self.applied)
        for key in ('connect', 'apply', 'prepare', 'export'):
            self.buttons[key].setEnabled(not busy and (connected or key == 'connect'))
        self.buttons['start'].setEnabled(self.state == 'ready')
        for widget in (self.joint, self.p_gain, self.d_gain, self.target, self.table):
            widget.setEnabled(not busy)
        for box in self.timings.values():
            box.setEnabled(not busy)

    def _open_plots(self):
        """Open a fresh installed-version PlotJuggler layout for selected joints."""
        if self.closing:
            return
        if self.process.state() != QProcess.NotRunning:
            self.plot_restart = True
            self.process.terminate()
            return
        executable = shutil.which('plotjuggler4')
        if executable is None:
            self.log.append('PlotJuggler 4 missing: install plotjuggler4 and its ROS 2 plugin.')
            return
        names = [name for name, check in self.checks.items() if check.isChecked()]
        layout = Path(self.directory.name) / 'selected.pj4.xml'
        write_layout(layout, names)
        environment = QProcessEnvironment.systemEnvironment()
        for name in ('QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            environment.remove(name)
        environment.insert('QT_QPA_PLATFORM', 'xcb')
        self.process.setProcessEnvironment(environment)
        self.process.start(executable, ['--nosplash'])
        self.log.append(f'After connecting the ROS subscriber, load layout: {layout}')

    def _copy_layout(self):
        """Copy the generated layout path for PlotJuggler's Load layout dialog."""
        layout = Path(self.directory.name) / 'selected.pj4.xml'
        names = [name for name, check in self.checks.items() if check.isChecked()]
        write_layout(layout, names)
        QApplication.clipboard().setText(str(layout))

    def _plots_finished(self, code, status):
        """Report plot errors or restart after a selection change.

        Args:
            code: Process exit code.
            status: Qt process exit status.
        """
        output = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        if not self.closing and code and not self.plot_restart:
            self.log.append(f'PlotJuggler exited ({code}, {status}): {output[-2000:]}')
        if self.plot_restart and not self.closing:
            self.plot_restart = False
            QTimer.singleShot(0, self._open_plots)

    def _plots_error(self, error):
        """Expose launch errors instead of claiming the plotter is running.

        Args:
            error: Qt process error.
        """
        if not self.closing:
            self.log.append(f'PlotJuggler process error {error}: {self.process.errorString()}')

    def shutdown_plugin(self):
        """Stop owned plotting/ROS resources without changing simulator physics."""
        self.closing = True
        self.timer.stop()
        self.session.quit.set()
        self.session.interrupt.set()
        self.session.join(timeout=3)
        if self.session.is_alive():
            self.log.append('ROS worker is still shutting down')
        if self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()
                self.process.waitForFinished(1000)
        self.directory.cleanup()


def main():
    """Start the panel as a standalone rqt application.

    Returns:
        rqt application exit status.
    """
    from rqt_gui.main import Main
    return Main().main(
        argv=sys.argv, standalone='humanoid_tuning.panel.TuningPanel',
        plugin_argument_provider=_plugin_arguments)


def _plugin_arguments(parser):
    """Register standalone rqt arguments before its strict initial parse.

    Args:
        parser: rqt argument parser.
    """
    parser.add_argument('--simulator', choices=['gazebo', 'mujoco'], default='gazebo')
