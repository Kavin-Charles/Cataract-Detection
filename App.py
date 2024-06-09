import cv2
import kivy.uix.screenmanager
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from Prediction import Predict
import time


def NextScreen(bool):
    print(bool)

class AppLayout(BoxLayout):
    def capture(self):

        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("Testing")
        hasCataract = Predict("Testing")
        NextScreen(hasCataract)

class TestApp(App):
    def build(self):
        return AppLayout()

TestApp().run()