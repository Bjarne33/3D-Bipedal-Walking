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
"""Generate a selected-joint PlotJuggler 4 layout with explicit units."""

import xml.etree.ElementTree as ET


SIGNALS = (
    ('Position (deg)', ('position_deg', 'reference_position_deg')),
    ('Velocity (deg/s)', ('velocity_deg_s', 'velocity_upper_deg_s', 'velocity_lower_deg_s')),
    ('Effort (N m)', ('effort_nm', 'effort_upper_nm', 'effort_lower_nm')),
)


def write_layout(path, joints):
    """Write a generic layout to apply after starting the ROS 2 subscriber.

    Args:
        path: Destination temporary layout path.
        joints: Selected joint names.
    """
    root = ET.Element('root', format='PlotJuggler', pj4_version='4', binding='generic')
    tabs = ET.SubElement(root, 'tabbed_widget', name='main', parent='main_window', id='')
    for name in joints:
        tab = ET.SubElement(tabs, 'Tab', containers='1', id=name, tab_name=name)
        container = ET.SubElement(tab, 'Container')
        splitter = ET.SubElement(
            container, 'DockSplitter', count='3', orientation='|', sizes='0.333;0.333;0.334')
        for index, (title, signals) in enumerate(SIGNALS):
            area = ET.SubElement(splitter, 'DockArea', id=f'{name}-{index}', name=title)
            plot = ET.SubElement(
                area, 'plot', id=f'{name}-{index}-plot', mode='TimeSeries',
                style='Steps', line_width='1.5', title=title, tracker_enabled='true')
            for signal, color in zip(signals, ('#0072b2', '#d55e00', '#009e73')):
                ET.SubElement(
                    plot, 'curve', topic=f'/humanoid_tuning/{name}/state', field=signal,
                    color=color, visible='true', markers_visible='false')
    ET.SubElement(tabs, 'currentTabIndex', index='0')
    ET.SubElement(root, 'left_panel_state', sources_tab='stream', streaming_buffer='30',
                  streaming_source='ROS 2 Topic Subscriber')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
