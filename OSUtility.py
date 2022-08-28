from .FileUtility import *
import  os
from tqdm import tqdm

class OSUtility:

  @staticmethod
  def getModulePlatform(filename):
       result = None
       ext = FileUtility.getFileExt(filename).lower()
       if (ext == 'dll' or ext == 'exe'):
         command = [ R'C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\14.28.29910\bin\Hostx64\x64\dumpbin.exe',   '/HEADERS', filename]
         p = subprocess.Popen(command, stdout=subprocess.PIPE)
         text = p.stdout.read()
         text = text.decode("utf-8")
         retcode = p.wait()
         list1 = text.split('\n')

         for l in list1:
           if l.find('machine (') != -1:
             if l.find('(x64)') != -1:
               result = 'x64'
               break
             else :
               result = 'x86'
               break


       return result

  @staticmethod
  def getFolderPlatform(src_path,dst_path = None):
    d_path = dst_path
    if d_path == None:
      d_path = os.path.join(src_path,'dependency.txt')

    x64 = []
    x86 = []

    files = FileUtility.getFolderModuleFiles(src_path)

    for i in tqdm(range(len(files)), ncols=100):
      file = files[i]
      platform = OSUtility.getModulePlatform(file)
      if platform != None:
        if platform == 'x64':
          x64.append(file)
        elif platform == 'x86':
          x86.append(file)

    with open(d_path,'w') as f:
      f.write('x64 :\n\n')
      for value in x64:
        f.write(value+'\n')

      f.write('x86 :\n\n')
      for value in x86 :
        f.write(value+'\n')

      f.close()







