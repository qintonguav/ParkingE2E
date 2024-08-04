#!/usr/bin/python
# coding:utf-8

from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *
import python_qt_binding.QtCore as QtCore

import sys

import rospy

from std_msgs.msg import Int32
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


from tf.transformations import euler_from_quaternion
import math


class UI(QWidget):
    def __init__(self):
        QWidget.__init__(self, None, QtCore.Qt.WindowStaysOnTopHint)
        rospy.init_node("PlanningUI")
        self.layout = QVBoxLayout()

        self._pub_parking_start = rospy.Publisher("/e2e_parking/start", Bool, queue_size=1)
        self._pub_parking_start_srv = rospy.ServiceProxy("/e2e_parking/srv_start", SetBool)
        self._pub_parking_finish = rospy.Publisher("/e2e_parking/finish", Bool, queue_size=1)
        self._pub_auto_drive = rospy.Publisher("/rock_can/auto_drive", Bool, queue_size=1)
        self._pub_release_manual_intervention = rospy.Publisher("/rock_can/release_manual_intervention", Bool, queue_size=1)

        self._pub_target_point_intervention = rospy.Publisher("/e2e_parking/set_target_point", Bool, queue_size=1)

        self._init_ui()
        print("Task UI is ready!!")


    def _init_ui(self):
        self._init_release_button()
        self._init_target_point_button()
        self._init_parking_label()
        self._init_auto_drive_button()
        self.setLayout(self.layout)


    def _init_release_button(self):
        row = QHBoxLayout()

        label = QLabel()
        label.setFixedSize(120, 30)
        label.setText("Release Manual:")
        row.addWidget(label)

        set_release_button = QPushButton()
        set_release_button.setText("Set")
        set_release_button.setFixedSize(120, 30)
        set_release_button.clicked.connect(self._release_manual)
        row.addWidget(set_release_button)

        self.layout.addLayout(row)

    def _init_target_point_button(self):
        row = QHBoxLayout()

        label = QLabel()
        label.setFixedSize(120, 30)
        label.setText("Set Target Point:")
        row.addWidget(label)

        set_target_point_button = QPushButton()
        set_target_point_button.setText("Set")
        set_target_point_button.setFixedSize(120, 30)
        set_target_point_button.clicked.connect(self._target_point)
        row.addWidget(set_target_point_button)

        self.layout.addLayout(row)



    def _init_parking_label(self):
        row = QHBoxLayout()

        label = QLabel(self)
        label.setFixedSize(120, 30)
        label.setText("Parking:")
        row.addWidget(label)

        button2 = QPushButton()
        button2.setText("Start")
        button2.setFixedSize(57, 30)
        button2.clicked.connect(self._parking_start_callback)
        row.addWidget(button2)

        button1 = QPushButton()
        button1.setText("End")  # 按钮名称
        button1.setFixedSize(57, 30)
        # 按下后触发on_save_button函数
        button1.clicked.connect(self._parking_end_callback)
        row.addWidget(button1)  # 将该按钮添加到该行(row)

        self.layout.addLayout(row)


    def _init_auto_drive_button(self):
        row = QHBoxLayout()

        label = QLabel()
        label.setFixedSize(120, 30)
        label.setText("Auto Drive:")
        row.addWidget(label)

        set_auto_drive_button = QPushButton()
        set_auto_drive_button.setText("Set")
        set_auto_drive_button.setFixedSize(57, 30)
        set_auto_drive_button.clicked.connect(self._set_auto_drive)
        row.addWidget(set_auto_drive_button)

        reset_auto_drive_button = QPushButton()
        reset_auto_drive_button.setText("Reset")
        reset_auto_drive_button.setFixedSize(57, 30)
        reset_auto_drive_button.clicked.connect(self._reset_auto_drive)
        row.addWidget(reset_auto_drive_button)

        self.layout.addLayout(row)


    def _parking_start_callback(self):


        self._pub_parking_start_srv.wait_for_service()
        state = Bool()
        state.data = True
        self._pub_parking_start.publish(state)
        resp = self._pub_parking_start_srv.call(True)
        if resp.success:
            print("OK!")

    def _parking_end_callback(self):
        state = Bool()
        state.data = True
        self._pub_parking_finish.publish(state)


    def _release_manual(self):
        flag = Bool()
        flag.data = True

        self._pub_release_manual_intervention.publish(flag)

    def _target_point(self):
        flag = Bool()
        flag.data = True

        self._pub_target_point_intervention.publish(flag)

    def _set_auto_drive(self):
        flag = Bool()
        flag.data = True

        self._pub_auto_drive.publish(flag)

    def _reset_auto_drive(self):
        flag = Bool()
        flag.data = False

        self._pub_auto_drive.publish(flag)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    planning_ui = UI()
    planning_ui.resize(100, 100)
    planning_ui.show()

    app.exec_()
