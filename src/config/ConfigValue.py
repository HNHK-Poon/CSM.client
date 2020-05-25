from configparser import ConfigParser
import os


class ConfigValue:
    def __init__(self):
        self.name = "ConfigValue"
        self.GENERAL = "GENERAL"
        self.parser = ConfigParser()
        print(os.getcwd())
        self.parser.read(os.getcwd() + "/config/config.ini")

    def get_value(self, section, field_name):
        return self.parser[section][field_name]
