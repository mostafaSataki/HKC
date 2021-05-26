
from .FileUtility import  *
from .Utility import  *
import sys


class ModuleUtility:

  @staticmethod
  def dumpbin():
    return R'C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.20.27508\bin\Hostx64\x64\dumpbin.exe',

  @staticmethod
  def getModuleExtensions():
    return ['exe', 'dll', 'lib', 'so', 'a']

  @staticmethod
  def checkIsModule(filename):
    ext = FileUtility.getFileExt(filename).lower()
    return ext in ModuleUtility.getModuleExtensions()

  @staticmethod
  def getFolderModuleFiles(path):
    result = []
    for (dirpath, dirnames, filenames) in os.walk(path):
      for filename in filenames:
        if ModuleUtility.checkIsModule(filename):
          result.append(os.path.join(dirpath, filename))

    return result

  @staticmethod
  def findFilePlatform(filename):
    if ModuleUtility.checkIsModule(filename) == False:
      return ''

    command = [ ModuleUtility.dumpbin(),  '/HEADERS', filename]
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    text = p.stdout.read()
    text = text.decode("utf-8")
    retcode = p.wait()
    list1 = text.split('\n')

    for l in list1:
      if l.find('machine (') != -1:

        if l.find('(x64)') != -1:
          return 'x64'
        else : return 'x86'

  @staticmethod
  def findPathPlatform(path):
    result = []
    files = ModuleUtility.getFolderModuleFiles(path)
    for file in files :
      result.append([file, ModuleUtility.findFilePlatform(file)])
    return result


  @staticmethod
  def savePathPlatform(path,filename):
    list = ModuleUtility.findPathPlatform(path)
    FileUtility.saveList2File(list,filename)

  @staticmethod
  def getFileDependency(filename):
    command = [ModuleUtility.dumpbin(), '/DEPENDENTS', filename]
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    text = p.stdout.read()
    result = []
    # text = text.decode('latin-1').encode("utf-8")
    try:
      text = text.decode("utf-8")
      retcode = p.wait()
      list1 = text.split('\n')


      for i,row1 in enumerate(list1):
        if row1.find('  Image has the following dependencies:') != -1:
          for j in range(i+2,len(list1)):
            if list1[j] != '\r':
              result.append(list1[j])
            else : break

    except UnicodeDecodeError:
       pass


    return result

  @staticmethod
  def getPathDependency(path):
    files = ModuleUtility.getFolderModuleFiles(path)
    result = {}
    for file in files :
      print(file)
      result[file] = ModuleUtility.getFileDependency(file)

    return result





  @staticmethod
  def savePathDependency(path,filename):
    data = ModuleUtility.getPathDependency(path)
    Utility.saveDict2File(data,filename)

  @staticmethod
  def savePathUniqueDependency(path,filename):
    data = ModuleUtility.getPathDependency(path)
    Utility.saveDictUniqueValues2File(data,filename)

  @staticmethod
  def getAppPath():
    if getattr(sys, 'frozen', False):
      return os.path.dirname(sys.executable)
    elif __file__:
      return os.path.dirname(__file__)

  @staticmethod
  def joinApp(name, path=None):
    path1 =  ModuleUtility.getAppPath()
    if (path):
      path1 = os.path.join(path1, path)

    return os.path.join(path1, name)





