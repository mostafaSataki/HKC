import re
from tqdm import tqdm
import string
from string import digits
from HKC import FileUtility
from HKC import Utility
import numpy as np
class StringUtility:
  @staticmethod
  def remove_substr(src_line, punc=True, number=True):
    numbers = r'[0-9]'
    result = src_line
    if number:
      result = re.sub(numbers, '', result)
    if punc:
      result = re.sub('[%s]' % re.escape(string.punctuation), '', result)

    return result

  @staticmethod
  def remove_substr_from_file(src_filename, dst_filename, punc=True, number=True, empty_line=True):
      with open(src_filename, 'r') as src_file:
        src_lines = src_file.readlines()
        dst_lines = []

        for src_line in src_lines:
          dst_line = StringUtility.remove_substr(src_line, punc, number)
          if empty_line:
            if dst_line != '\n':
              dst_lines.append(dst_line)
          else:
            dst_lines.append(dst_line)

      with open(dst_filename, 'w') as dst_file:
        dst_file.writelines(dst_lines)
        dst_file.close()

  @staticmethod
  def removeDuplicateLines(src_filename,dst_filename):
    with open(src_filename,'r') as src_file:
      src_lines = src_file.readlines()

    dst_lines = list(set(src_lines))
    with open(dst_filename,'w') as dst_file:
      dst_file.writelines(dst_lines)

  @staticmethod
  def line2words(src_line):
    src_line = src_line.split('\n')[0]
    return src_line.split(' ')

  @staticmethod
  def lines2words(src_lines,unique = True):
    dst_lines = []
    if unique :
      result = set()
      for src_line in src_lines:
        words = StringUtility.line2words(src_line)
        for word in words:
          result.add(word)

      for item in result:
        dst_lines.append(item+'\n')

    else :
      for src_line in src_lines:
        words = StringUtility.line2words(src_line)
        for word in words:
          dst_lines.append(word + '\n')

    return dst_lines



  @staticmethod
  def line2wordsFiles(src_filename,dst_filename,unique =True):
       with open(src_filename, 'r') as src_file:
         src_lines = src_file.readlines()
         dst_lines = StringUtility.lines2words(src_lines,unique)
       with open(dst_filename, 'w') as dst_file:
         dst_file.writelines(dst_lines)
         dst_file.close()

  @staticmethod
  def mergeTextFiles(src_path,dst_filename):
     files = FileUtility.getFolderFiles(src_path,'txt')

     with open(dst_filename,'w') as dst_file:
       for i in tqdm(range(len(files)), ncols=100):
          file = files[i]
          with open(file,'r') as src_file:
            lines = src_file.readlines()
            dst_file.writelines(lines)

          dst_file.flush()

  @staticmethod
  def columnize_text(src_filename, dst_filename, maxchar_in_row):
    with open(src_filename, 'r') as src_file:
      src_lines = src_file.readlines()
      dst_lines = []
      cur_line = ""
      for src_line in src_lines:
        s_line = src_line.split('\n')[0] + ' '
        line_len = len(cur_line) + len(s_line)
        if line_len >= maxchar_in_row:
          dst_lines.append(cur_line + '\n')
          cur_line = s_line
        else:
          cur_line += s_line
    with open(dst_filename, 'w') as dst_file:
      dst_file.writelines(dst_lines)

  @staticmethod
  def removeBlankLines(src_filename, dst_filename):
    with open(src_filename, 'r') as src_file:
      src_lines = src_file.readlines()
      dst_lines = []
      for src_line in src_lines:
        s_line = src_line.rstrip('\n')
        if len(s_line):
          dst_lines.append(src_line)

      with open(dst_filename, 'w') as dst_file:
        dst_file.writelines(dst_lines)


  @staticmethod
  def splitLines(src_filename, dst_train_filename, dst_val_filename, train_per):
    with open(src_filename, 'r') as src_file:
      src_lines = src_file.readlines()
      l = np.arange(0, len(src_lines))
      np.random.shuffle(l)

      train_count = int(len(src_lines) * train_per)
      train_lines = []
      val_lines = []
      for i in range(len(src_lines)):
        index = l[i]
        if i < train_count:
          train_lines.append(src_lines[index])
        else:
          val_lines.append(src_lines[index])

      with open(dst_train_filename, 'w') as dst_train_file:
        dst_train_file.writelines(train_lines)

      with open(dst_val_filename, 'w') as dst_val_file:
        dst_val_file.writelines(val_lines)


  @staticmethod
  def parseSqlFile(sql_filename,colum_id,dst_filename =None):
    result = []
    with open(sql_filename,'r') as src_file:
      lines = src_file.readlines()
      for line in lines:
        if line.startswith('INSERT INTO'):
          start = line.find('(')
          end = line.find(')')
          columns_str = line[start+1:end]
          columns = columns_str.split(',')
          column_str = str(columns[colum_id])
          column_str = StringUtility.remove_substr(column_str)
          result.append(column_str+'\n')
    if dst_filename:
      with open(dst_filename,'w') as dst_file:
        dst_file.writelines(result)
    return result

  @staticmethod
  def addYeh2Word(src_filename,dst_filename):
    with open(src_filename) as src_file:
      src_lines = src_file.readlines()

    dst_lines = []
    for src_line in src_lines:
      dst_line = src_line.rstrip()
      if not dst_line.endswith('ی'):
        dst_line +='ی'
      dst_line +='\n'
      dst_lines.append(dst_line)

    dst_lines.extend(src_lines)
    with open(dst_filename,'w') as dst_file:
      dst_file.writelines(dst_lines)

  @staticmethod
  def removeLinesContains(src_filename,dst_filename,words):
    with open(src_filename,'r') as src_file:
      src_lines = src_file.readlines()

    dst_lines = []
    for src_line in src_lines:
      flag = False
      for word in words:
        if word in src_line:
          flag = True
          break
      if flag == False:
        dst_lines.append(src_line)

    with open(dst_filename,'w') as dst_file:
      dst_file.writelines(dst_lines)

  @staticmethod
  def shuffleLines(src_filename,dst_filename):
    with open(src_filename,'r') as src_file:
      src_lines = src_file.readlines()

    dst_lines = []
    rl = Utility.getRandomList(len(src_lines))
    for item in rl:
      dst_lines.append(src_lines[item])

    with open(dst_filename,'w') as dst_file:
      dst_file.writelines(dst_lines)



  def clean_latin(src_filename,dst_filename):
    dst_lines = []
    with open(src_filename,'r') as src_file:
      src_lines = src_file.readlines()
    for src_line in src_lines:
      flag = True
      for ch in src_line:
        if (ch >='a' and ch <= 'z') or (ch >='A' and ch <= 'Z'):
          flag = False
          break
      if flag:
        dst_lines.append(src_line)
    with open(dst_filename,'w') as dst_file:
      dst_file.writelines(dst_lines)















