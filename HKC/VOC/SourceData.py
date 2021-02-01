from xml.etree import ElementTree

class SourceData:
    @property
    def database(self):
      return self._database

    @database.setter
    def database(self, value):
      self._database = value

    def __init__(self, value="Unknown"):
      self._database = value


    def read(self,data):
        self._database = data.find('database').text

    def write(self,data):
        ElementTree.SubElement(data, "database").text = self._database
