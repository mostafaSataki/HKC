import  cv2
import  numpy as np
from .FileUtility import *
import  os
import cv2
import time
from tqdm import tqdm
import math
import subprocess
from scipy.spatial import distance as dist
from  .Utility import *
from multiprocessing import Pool
from PIL import Image, ImageSequence
import imutils
 
class CvUtility:

  @staticmethod
  def resizeImage(src_image,size,interpolation = None):

      scale = (float(size[0] / src_image.shape[1] ) , float(size[1]) / src_image.shape[0]  )

      dst_image = cv2.resize(src_image,size,interpolation=interpolation)
      return dst_image,scale

  @staticmethod
  def resizeDown(src_image, size, interpolation=cv2.INTER_LINEAR):
    src_width = src_image.shape[1]
    src_height = src_image.shape[0]

    dst_image = src_image.copy()
    dst_width = dst_image.shape[1]
    dst_height = dst_image.shape[0]
    while (dst_width / 2 > size[0] and dst_height / 2 > size[1]):
      dst_image = cv2.pyrDown(dst_image, (int(dst_width / 2), int(dst_height / 2)))
      dst_width = dst_image.shape[1]
      dst_height = dst_image.shape[0]

    if (dst_width != size[0] and dst_height != size[1]):
      dst_image = cv2.resize(dst_image, size, 0, 0, interpolation)

    out_size = (float(src_width) / size[0], float(src_height) / size[1])
    return dst_image, out_size

  @staticmethod
  def imreadU(filename,flag):
    return cv2.imdecode(np.fromfile(filename,dtype=np.uint8), flag)


  @staticmethod
  def loadImage(filename,flag = cv2.IMREAD_COLOR):

    if FileUtility.checkIsImage(filename):
      image = CvUtility.imreadU(filename,flag)
      if not  image is None:
        if flag != cv2.IMREAD_COLOR:
           image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
      return image

  @staticmethod
  def imwriteU(filename,image, params=None):
    is_success, im_buf_arr = cv2.imencode(".jpg", image,params)
    im_buf_arr.tofile(filename)

  @staticmethod
  def imwrite_branch(img,dst_path,src_filename,dst_branch ='exp'):
    dst_path_branch = FileUtility.get_next_path(dst_path, dst_branch)

    if not os.path.exists(dst_path_branch):
      os.makedirs(dst_path_branch)

    dst_filename = FileUtility.getDstFilename(src_filename, dst_path_branch)
    cv2.imwrite(dst_filename,img)
    return   dst_filename

  @staticmethod
  def changeImageFormat(src_filename,dst_filename,jpeg_quality = 30):
    image = CvUtility.loadImage(src_filename)
    if not image is None:
      CvUtility.imwriteU(dst_filename,image,[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    # cv2.imwrite(dst_filename,image)

  def changeDownImageFormat(args, src_filename, dst_filename, jpeg_quality=60):
    image = CvUtility.loadImage(src_filename)
    if image is None:
      print('error None image' + src_filename)
      return

    image, _ = self.resizeDown(image, (args.input_size, args.input_size), cv2.INTER_CUBIC)
    if not image is None:
      CvUtility.imwriteU(dst_filename, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

  @staticmethod
  def changeImagesFormat(src_path,dst_path, dst_ext, jpeg_quality = 60, delete_source = False):
    FileUtility.copyFullSubFolders(src_path,dst_path)

    src_filenames = FileUtility.getFolderImageFiles(src_path)
    dst_filenames = FileUtility.getDstFilenames2(src_filenames,dst_path,src_path )


    dst_filenames = FileUtility.changeFilesExt(dst_filenames, dst_ext)

    for i in tqdm(range(len(dst_filenames)), ncols=100):
       # print(src_filenames[i])
       dst_filename = dst_filenames[i]
       # print(i/len(dst_filenames)* 100, dst_filename)
       CvUtility.changeImageFormat(src_filenames[i],dst_filename,jpeg_quality)
       if delete_source:
         os.remove(src_filenames[i])

  @staticmethod
  def change_filename_to_foldername(src_path):

    src_files = FileUtility.getFolderImageFiles(src_path)
    for i in tqdm(range(len(src_files)), ncols=100):
      src_file = src_files[i]
      f = FileUtility.get_file_upfolder(src_file)
      f_name = FileUtility.changeFileName(src_file, f)
      os.rename(src_file, f_name)

  @staticmethod
  def getImageSize(src):
    return [src.shape[0], src.shape[1]]


  @staticmethod
  def getImageRect(filename):
    image = CvUtility.imwriteU(filename,1)
    return (0,0,image.shape[1], image.shape[0])

  @staticmethod
  def getImageRect(image):
    return (0,0,image.shape[1], image.shape[0])


  @staticmethod
  def getRectSize(rect):
    return rect[2], rect[3]

  @staticmethod
  def unionRects(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

  @staticmethod
  def findRects(src):
     contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     result = []
     for i in range(len(contours)):
        if hierarchy[0][i][3] == -1 :
           r = cv2.boundingRect(contours[i])
           result.append(r)

     return result

  @staticmethod
  def findRect(src):
    rects = CvUtility.findRects(src)
    if len(rects):
      result = rects[0]
      for i in range(len(rects) -1) :
        result = CvUtility.unionRects(result,rects[i+1])
      return result
    else : return (-1,-1,0,0)

  @staticmethod
  def cropImage(src):
      r = CvUtility.findRect(src)
      x,y,w,h = r
      return  src[y:y+h,x:x+w]

  @staticmethod
  def crop_roi(image,roi):
    x, y, w, h = roi
    return image[y:y + h, x:x + w]

  @staticmethod
  def crop_rois(image, rois):
      result = []
      for roi in rois:
        result.append(CvUtility.crop_roi(image,roi))
      return result


  @staticmethod
  def getChannel(src):
    count = len(src.shape)
    if count == 3:
      return src.shape[2]
    else:
      return 1

  @staticmethod
  def batchResize(path,size):
    files = FileUtility.getFolderImageFiles(path)
    for file in files:
      img = cv2.imread(file,1)
      img = cv2.resize(img,dsize=size,interpolation=cv2.INTER_CUBIC)
      cv2.imwrite(file,img)

  @staticmethod
  def fitOnSizeCorrect(src, d_size):

    s_a = float(d_size[0]) / src.shape[1]
    s_b = float(d_size[1]) / src.shape[0]
    size_a = (int(src.shape[1] * s_a), int(src.shape[0] * s_a))
    size_b = (int(src.shape[1] * s_b), int(src.shape[0] * s_b))

    if (size_a[0] <= d_size[0] and size_a[1] <= d_size[1]):
      dst_size = size_a
      scale = s_a
    elif (size_b[0] <= d_size[0] and size_b[1] <= d_size[1]):
      dst_size = size_b
      scale = s_b


    return dst_size,scale

  @staticmethod
  def copyCenter(patch,dst):

    patch_shape = patch.shape
    dst_shape = dst.shape

    x = int( (dst_shape[1] - patch_shape[1]) /2)
    y = int( (dst_shape[0] - patch_shape[0]) /2)
    dst[y:y+patch_shape[0],x:x+patch_shape[1]] = patch

    return dst

  @staticmethod
  def get_size_order(ref_size):
    # Determine the order of ref_size
    if ref_size[0] < ref_size[1]:
      order = "Asc"
    else:
      order = "Desc"
    return order

  @staticmethod
  def autoDimSize(size, ref_size):
    order = CvUtility.get_size_order(ref_size)
    # Convert size to a list, sort it, and convert it back to a tuple
    size = list(size)
    if order == "Asc":
      size.sort()
    else:
      size.sort(reverse=True)
    size = tuple(size)
    return size
  @staticmethod
  def fitOnSizeMat(src, dst_size, interpolation=None,auto_dim = False):

    if auto_dim:
        src_size = CvUtility.getImageSize(src)
        dst_size = CvUtility.autoDimSize(dst_size,src_size)

    size, scale = CvUtility.fitOnSizeCorrect(src, dst_size)
    result = cv2.resize(src, size, 0, 0)
    if len(result.shape) == 2:
      result = result.reshape(size[1], size[0], 1)

    return result, scale

  @staticmethod
  def fitOnSizeMatStatic(src, dst_size,color, interpolation=None):
    dst_image,scale = CvUtility.fitOnSizeMat(src,dst_size,interpolation)

    channel = 1
    if len(src.shape) == 3:
      channel =src.shape[2]

    dst_image_static = np.ones((dst_size[0],dst_size[1],channel),np.uint8)
    dst_image_static[:, :] = 255
    CvUtility.copyCenter(dst_image,dst_image_static)
    return dst_image_static

  @staticmethod
  def fitOnSizeBatch(src_path, dst_path, dst_size,jpeg_quality = None,interpolation = cv2.INTER_LINEAR):
    FileUtility.copyFullSubFolders(src_path,dst_path)
    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files,src_path, dst_path)

    for i in tqdm(range(len(src_files)), ncols=100):
      src_file = src_files[i]
      dst_file = dst_files[i]
      src_image = cv2.imread(src_file)
      dst_image, scale = CvUtility.fitOnSizeMat(src_image, dst_size,interpolation)
      if jpeg_quality :
        cv2.imwrite(dst_file, dst_image,[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
      else :cv2.imwrite(dst_file, dst_image,)
        

  @staticmethod
  def fitOnSizeStaticBatch(src_path, dst_path, dst_size,color =(255,255,255),jpeg_quality = 30,interpolation = cv2.INTER_LINEAR):
    FileUtility.copyFullSubFolders(src_path, dst_path)

    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files,src_path, dst_path)

    for i in tqdm(range(len(src_files)), ncols=100):
      src_file = src_files[i]

      dst_file = dst_files[i]
      src_image = cv2.imread(src_file)
      # print( int(i / len(src_files) * 100))

      dst_image = CvUtility.fitOnSizeMatStatic(src_image,dst_size,color,interpolation)
      cv2.imwrite(dst_file, dst_image,[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])



  @staticmethod
  def resizeBatch(src_path,dst_path,dst_size, interpolation=None,jpeg_quality = 30):


    src_files = FileUtility.getFolderImageFiles(src_path)
    if src_path != dst_path :
      FileUtility.create_folder_if_not_exists(dst_path)
      FileUtility.copyFullSubFolders(src_path,dst_path   )

    dst_files = FileUtility.getDstFilenames2(src_files,dst_path,src_path )

    for i in tqdm(range(len(src_files)), ncols=100):
      src_file = src_files[i]
      dst_file = dst_files[i]
      src_image = cv2.imread(src_file)
      dst_image = cv2.resize(src_image, dst_size,interpolation)
      cv2.imwrite(dst_file,dst_image,[cv2.IMWRITE_JPEG_QUALITY,jpeg_quality])

  @staticmethod
  def resize(src_path,dst_path,dst_size,coefficient =None, post_fix = "",jpeg_quality = 70):

    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files,src_path,dst_path)

    FileUtility.copyFullSubFolders(src_path,dst_path)

    if post_fix != "":
       dst_files = FileUtility.changeFilesnamePostfix(dst_files,post_fix)

    for i in tqdm(range( len(src_files)), ncols=100):
      src_filename = src_files[i]
      dst_filename = dst_files[i]

      src_image = cv2.imread(src_filename)

      new_size = dst_size
      if coefficient :
        new_width = int(src_image.shape[1] * coefficient)
        new_height = int(src_image.shape[0] * coefficient)
        new_size = (new_width,new_height)

      dst_image = cv2.resize(src_image,new_size,cv2.INTER_CUBIC)

      cv2.imwrite(dst_filename,dst_image,[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

  @staticmethod
  def fitOnSizeMatBatch(src_path, dst_path, dst_size, interpolation=None, dst_ext = None, post_fix="", jpeg_quality=70
                        ,auto_dim = False):


    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files,dst_path, src_path )

    FileUtility.copyFullSubFolders(src_path, dst_path)

    if post_fix != "":
      dst_files = FileUtility.changeFilesnamePostfix(dst_files, post_fix)

      if dst_ext is not None:
         dst_files = FileUtility.changeFilesExt(dst_files, dst_ext)

    for i in tqdm(range(len(src_files)), ncols=100):
      src_filename = src_files[i]
      dst_filename = dst_files[i]

      src_image = cv2.imread(src_filename)

      dst_image,_ = CvUtility.fitOnSizeMat(src_image,dst_size,interpolation,auto_dim)

      CvUtility.imwriteU(dst_filename, dst_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


  @staticmethod
  def resizeGray(src_path,dst_path,dst_size, post_fix = "",jpeg_quality = 70):
    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files,src_path,dst_path)

    FileUtility.copyFullSubFolders(src_path,dst_path)

    if post_fix != "":
       dst_files = FileUtility.changeFilesnamePostfix(dst_files,post_fix)

    for i in tqdm(range( len(src_files)), ncols=100):
      src_filename = src_files[i]
      dst_filename = dst_files[i]

      src_image = cv2.imread(src_filename,0)
      dst_image = cv2.resize(src_image,dst_size,cv2.INTER_CUBIC)

      cv2.imwrite(dst_filename,dst_image,[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

  @staticmethod
  def toGray(src_path,dst_path, post_fix = "",jpeg_quality = 30):
    src_files = FileUtility.getFolderImageFiles(src_path)
    if src_path != dst_path :
      dst_files = FileUtility.getDstFilenames2(src_files,src_path,dst_path)
      FileUtility.copyFullSubFolders(src_path, dst_path)
    else : dst_files = src_files



    if post_fix != "":
       dst_files = FileUtility.changeFilesnamePostfix(dst_files,post_fix)

    for i in tqdm(range(len(src_files)), ncols=100):
      src_filename = src_files[i]
      dst_filename = dst_files[i]

      src_image = cv2.imread(src_filename)

      dst_image = cv2.cvtColor(src_image,cv2.COLOR_BGR2GRAY);

      cv2.imwrite(dst_filename,dst_image,[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


  @staticmethod
  def resizeScaleBatch(src_path,dst_path,scale):
    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames(src_files, dst_path)

    for i, src_file in enumerate(src_files):
      print(src_file)
      dst_file = dst_files[i]
      src_image = cv2.imread(src_file)

      dst_image = cv2.resize(src_image,None,fx =scale[0],fy=scale[1])
      cv2.imwrite(dst_file,dst_image)

  @staticmethod
  def expandRect(rect,expand_size):
    half_size = (int(expand_size[0] / 2),int(expand_size[1] / 2))
    return (rect[0] - half_size[0],rect[1] - half_size[1],rect[2] +expand_size[0] , rect[3]+ expand_size[1])

  @staticmethod
  def expandRects(rects,expand_size):
    result = []
    for rect in rects:
      result.append(CvUtility.expandRect(rect,expand_size))
    return result

  @staticmethod
  def getSizeCofi(size,cofi):
    width =  size[0] * cofi
    height = size[1] * cofi
    return [int(width),int(height)]

  @staticmethod
  def getScale(size1,size2):
    return (size1[0] / size2[0],size1[1] / size2[1])

  @staticmethod
  def scaleRect(rect,scale):
    return [int(rect[0] * scale[0]),int(rect[1] * scale[1]),int(rect[2] * scale[0]),int(rect[3] * scale[1])]

  @staticmethod
  def scaleRects(rects, scale):
    result = []
    for rect in rects:
      result.append(CvUtility.scaleRect(rect,scale))
    return result

  @staticmethod
  def expandRectCofi(rect,expand_cofi):
    rect_size = CvUtility.getRectSize(rect)
    expand_size = CvUtility.getSizeCofi(rect_size,expand_cofi)
    half_size = [int(expand_size[0] / 2),int(expand_size[1] / 2)]
    return [rect[0] - half_size[0],rect[1] - half_size[1],rect[2] +expand_size[0] , rect[3]+ expand_size[1]]


  @staticmethod
  def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)
  @staticmethod
  def colorMask(image,lower,upper):
    return cv2.inRange(image,np.array(lower),np.array(upper))

  @staticmethod
  def getMaskRegion(mask):


    cnts,h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i ,c in enumerate(cnts):
      r = cv2.boundingRect(c)
      if i == 0:
        result = r
      else :  result = CvUtility.unionRects(result,r)


    return result

  @staticmethod
  def drawRect(image,region,color=(0,255,0),thickness=1):
    cv2.rectangle(image,(region[0],region[1]),(region[0]+ region[2],region[1]+ region[3]),color,thickness)

  @staticmethod
  def getMatRect(image):
    h,w = image.shape
    return [0,0,w,h]

  @staticmethod
  def breakRect(region,cells_count):
    l = region[0]
    t = region[1]
    w = region[2] // cells_count[0]
    h = region[3] // cells_count[1]
    result =[]
    for i in range(cells_count[1]):

      for j in range(cells_count[0]):
        r = [l+j*w,t+ i * h, w, h]
        result.append(r)

    return result


  @staticmethod
  def drawRects(image, regions,color=(0,255,0),thickness = 1):
    for r in regions:
      CvUtility.drawRect(image,r,color,thickness)

  @staticmethod
  def getPatch(image,r):
    x,y,width,height = r
    return image[y:y + height, x:x + width]

  @staticmethod
  def savePatchs(image,regions,dst_path,filename,counter ,ext = '.jpg'):

    for i,region in enumerate(regions):
      f_name = os.path.join(dst_path,filename+str(counter+i)+ext)
      patch = CvUtility.getPatch(image,region)
      cv2.imwrite(f_name,patch)

    return counter + len(regions)

  @staticmethod
  def augmentData(src_path, dst_path, seq, aug_count=10, postfix=''):
    src_image_filenames = FileUtility.getFolderImageFiles(src_path)

    dst_image_filenames = FileUtility.getDstFilenames(src_image_filenames, dst_path, postfix)
    if aug_count > 0:
      all_images = []
      all_dst_images_filename = []
      all_dst_jsons_filename = []
      all_kps = []
      all_src_json_filenames = []

      sub_folders = FileUtility.getSubfolders(src_path)
      FileUtility.createSubfolders(dst_path,sub_folders)

      for i, src_image_filename in enumerate(src_image_filenames):
        # print(src_image_filename)
        src_image = cv2.imread(src_image_filename)
        dst_image_filename = dst_image_filenames[i]

        images_filename = FileUtility.addCounters2Filename(dst_image_filename,0,aug_count)


        for j in range(aug_count):
          all_images.append(src_image)

        all_dst_images_filename += images_filename

      all_images_aug = seq(images=all_images)

      for i,cur_image in enumerate(all_images_aug):
        cv2.imwrite(all_dst_images_filename[i],all_images_aug[i], [int(cv2.IMWRITE_JPEG_QUALITY), 70])



    FileUtility.copyFilesByName(src_image_filenames,dst_image_filenames)




  @staticmethod
  def augmentImages(src_path,dst_path,seq, group_size = 200,postfix =''):

    FileUtility.copyFullSubFolders(src_path,dst_path)

    sub_folders = FileUtility.getSubfolders(src_path)

    for i in tqdm(range(len(sub_folders)), ncols=100):
      sub_folder = sub_folders[i]
      src_group = os.path.join(src_path,sub_folder)
      dst_group = os.path.join(dst_path, sub_folder)
      src_filenames = FileUtility.getFolderImageFiles(src_group)
      count = 0
      if len(src_filenames) < group_size:
        count = math.ceil ( (group_size - len(src_filenames)) / len(src_filenames))

      CvUtility.augmentData(src_group,dst_group,seq,count,postfix)


  @staticmethod
  def extractVideoFileFrames(video_filename, dst_path, ext ='.jpg', file_prefix = None, ffmpeg = None):
    if ffmpeg is None:
       ffmpeg_file = r'E:\Source\Repo\ffmpeg\bin\ffmpeg.exe'
    else : ffmpeg_file = ffmpeg

    if file_prefix :
        fname = file_prefix
    else : fname = FileUtility.getFilenameWithoutExt(video_filename)

    if not os.path.isdir(dst_path):
        os.makedirs(dst_path,exist_ok=True)

    command_str = ffmpeg_file+' -i '+'"'+ video_filename+ '" "'+os.path.join(dst_path,fname+"%03d"+ext)+'"'

    command = [ffmpeg_file,'-i',video_filename,os.path.join(dst_path,fname+"%03d"+ext)]

    p = subprocess.Popen(command_str, shell=False,  stdout = subprocess.PIPE, stderr = subprocess.DEVNULL)

  @staticmethod
  def _extractVideoPathFrames(src_path, dst_path, ext='.jpg', file_prefix=None, ffmpeg=None):
    if not os.path.isdir(dst_path):
      os.makedirs(os.path.join(dst_path))
    files_name = FileUtility.getFolderFiles(src_path, False, False)
    full_files_name = FileUtility.getFolderFiles(src_path, False,True)
    sub_folders = FileUtility.createSubfolders(dst_path, files_name)
    for i, file in enumerate(full_files_name):
      cur_sub_folder = sub_folders[i]
      fname = os.path.join(src_path,file)
      CvUtility.extractVideoFileFrames(fname, cur_sub_folder, ext, file_prefix, ffmpeg)

  @staticmethod
  def extractVideoPathFrames(src_path, dst_path, ext='.jpg', file_prefix=None, ffmpeg=None):

    folders = FileUtility.getSubfolders(src_path)
    if len(folders) == 0:
      CvUtility._extractVideoPathFrames(src_path, dst_path, ext, file_prefix, ffmpeg)
    else:
      # folders = FileUtility.createSubfolders(dst_path, folders)
      dst_sub_folders = FileUtility.createSubfolders(dst_path, folders)

      for i, fld in enumerate(folders):
        src_sub_folder =os.path.join(src_path, fld)
        CvUtility._extractVideoPathFrames(src_sub_folder, dst_sub_folders[i], ext, file_prefix, ffmpeg)

  @staticmethod
  def recompressImages(path,jpeg_quality = 30,remain_orginal = False):
    image_files = FileUtility.getFolderImageFiles(path)

    for i in tqdm(range(len(image_files)), ncols=100):
      image_file = image_files[i]
      image = cv2.imread(image_file,1)
      cur_ext = FileUtility.getFileExt(image_file)
      image_new_file = image_file
      if cur_ext.lower() != 'jpg':
        image_new_file = FileUtility.changeFileExt(image_file,'jpg')
        if remain_orginal == False:
          os.remove(image_file)

      cv2.imwrite(image_new_file,image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

  @staticmethod
  def convertFiles2Folders(src_path, dst_path):
    if not os.path.exists(dst_path):
      os.makedirs(dst_path)

    all_name = glob.glob(src_path + '/*')
    for name in all_name:
      if os.path.isdir(name):
        n = name[len(src_path) + 1:]
        convertFiles2Folders(name, os.path.join(dst_path, n))

      else:
        tokens = FileUtility.getFileTokens(name)
        dst = os.path.join(dst_path, tokens[1])
        if not os.path.exists(dst):
          os.makedirs(dst)

  def getPSNR(I1, I2):
    s1 = cv2.absdiff(I1, I2)  # |I1 - I2|
    s1 = np.float32(s1)  # cannot make a square on 8 bits
    s1 = s1 * s1  # |I1 - I2|^2
    sse = s1.sum()  # sum elements per channel
    if sse <= 1e-10:  # sum channels
      return 0  # for small values return zero
    else:
      shape = I1.shape
      mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
      psnr = 10.0 * np.log10((255 * 255) / mse)
      return psnr

  @staticmethod
  def imageROI(image,region):
    return image[region[1]:region[1] + region[3],region[0]:region[0] + region[2] ]

  @staticmethod
  def getPSNRRegions(image1,region1, image2,region2):
    union_region = CvUtility.unionRects(region1,region2)
    return CvUtility.getPSNR(CvUtility.imageROI(image1,union_region),CvUtility.imageROI(image2,union_region))


  @staticmethod
  def clearSimilarFrames(src_path,similarity_thresh = 21):
    image_filesname = FileUtility.getFolderImageFiles(src_path)
    ref_id = 0
    ref_image_filename = image_filesname[ref_id]
    ref_img = cv2.imread(ref_image_filename)

    for i in tqdm(range(len(image_filesname)), ncols=100):
      cur_img = cv2.imread(image_filesname[i])
      psnr = CvUtility.getPSNR(ref_img, cur_img)
      if psnr < similarity_thresh:
        ref_image_filename = image_filesname[i]
        ref_img = cv2.imread(ref_image_filename)
      else : os.remove(image_filesname[i])

  # [get-mssim]
  def getMSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS

    I1 = np.float32(i1)  # cannot calculate on one byte large values
    I2 = np.float32(i2)

    I2_2 = I2 * I2  # I2^2
    I1_2 = I1 * I1  # I1^2
    I1_I2 = I1 * I2  # I1 * I2
    # END INITS

    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2

    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2

    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    ssim_map = cv.divide(t3, t1)  # ssim_map =  t3./t1;

    mssim = cv.mean(ssim_map)  # mssim = average of ssim map
    return mssim
  # [get-mssim]

  @staticmethod
  def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


  def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0: return ()  # or (0,0,0,0) ?
    return (x, y, w, h)


  @staticmethod
  def rectsDistance(region1,region2):
    return dist.euclidean(CvUtility.getRectCenter(region1) , CvUtility.getRectCenter(region2))

  @staticmethod
  def rectArea(region):
    return region[2] * region[3]

  @staticmethod
  def rectsDimDiff(region1,region2):
     w = abs(region1[2] - region2[2])
     h = abs(region1[3] - region2[3])
     return max(w,h)

  @staticmethod
  def similariyRects(region1,region2,distance_thresh = 100,dim_thresh = 100):
     rects_distance = CvUtility.rectsDistance(region1,region2)
     area_diff = CvUtility.rectsDimDiff(region1,region2)

     return rects_distance < distance_thresh and area_diff < dim_thresh

  
  @staticmethod
  def similariy(image1,region1,image2,region2,psnr_thresh = 19,distance_thresh = 100,area_thresh = 100):
     result = CvUtility.similariyRects(region1,region2,distance_thresh,area_thresh)
     if result:
        psnr = CvUtility.getPSNRRegions(image1,region1,image2,region2)
        # print("psnr ",psnr)
        result = psnr > psnr_thresh 
     
     return result

  @staticmethod
  def flipHorz(src_path,dst_path,post_fix = ""):
    FileUtility.copyFullSubFolders(src_path,dst_path)
    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files,src_path,dst_path)
    if post_fix != "":
      dst_files = FileUtility.changeFilesnamePostfix(dst_files,post_fix)

    for i in tqdm(range(len(src_files)), ncols=100):
       src_file = src_files[i]
       dst_file = dst_files[i]

       src_image = cv2.imread(src_file)
       dst_image =  cv2.flip(src_image,1)

       cv2.imwrite(dst_file,dst_image,[cv2.IMWRITE_JPEG_QUALITY, 30])



  @staticmethod
  def readImage(filename,size = None,norm = False,gray = False,float_type = True):
        if gray:
           image = cv2.imread(filename, 0)
           channel = 1
        else :
            image = cv2.imread(filename, 1)
            channel = 3

        if np.shape(image) == ():
            None

        if float_type :
           image = image.astype('f4')

        if norm :
           image = image / 255.0

        if size :
           image = image.reshape(size[1], size[0], channel)

        return image

  @staticmethod
  def readImages(filenames,size = None,norm = False,gray = False,float_type = True):
    result = []
    for i in tqdm(range(len(filenames)), ncols=100):
      filename = filenames[i]
      result.append(CvUtility.readImage(filename,size,norm,gray,float_type))
    return result

  # @staticmethod
  # def resizeDown(src_image,  size, interpolation ) :
  # 
  #   dst_image = src_image.copy()
  #   shape = dst_image.shape
  #   w = shape[1]
  #   h = shape[0]
  # 
  #   while (w / 2 > size[0]) and ( h / 2 > size[1]) :
  #     dst_image =cv2.pyrDown(dst_image,(int(w / 2), int(h / 2)))
  #     shape = dst_image.shape
  #     w = shape[1]
  #     h = shape[0]
  # 
  #   shape = dst_image.shape
  #   w = shape[1]
  #   h = shape[0]
  # 
  #   if (w != size[0] or h != size[1]):
  #      dst_image = cv2.resize(dst_image,  size, 0, 0, interpolation);
  # 
  #   return  dst_image

  @staticmethod
  def resizeDownBatch(src_path,dst_path,dst_size, interpolation=cv2.INTER_LINEAR,jpeg_quality = 30):


    src_files = FileUtility.getFolderImageFiles(src_path)
    if src_path != dst_path :
      FileUtility.copyFullSubFolders(src_path,dst_path
                                   )
    dst_files = FileUtility.getDstFilenames2(src_files, dst_path,src_path)
    dst_files = FileUtility.changeFilesExt(dst_files,'jpg')

    for i in tqdm(range(len(src_files)), ncols=100):
      src_file = src_files[i]
      dst_file = dst_files[i]
      src_image = cv2.imread(src_file)
      dst_image = CvUtility.resizeDown(src_image, dst_size,interpolation)
      cv2.imwrite(dst_file,dst_image,[cv2.IMWRITE_JPEG_QUALITY,jpeg_quality])


  @staticmethod
  def loadImages(files,gray = False):
    images = []
    flag = 1
    if gray :
      flag = 0

    for i in tqdm(range(len(files)), ncols=100):
        file = files[i]
        images.append(cv2.imread(file,flag))


    return images



 
  @staticmethod
  def imshowScale(title,image,size):

    d_size,_ = CvUtility.fitOnSizeCorrect(image,size)
    d_image = cv2.resize(image,d_size)
    cv2.imshow(title,d_image)

  @staticmethod
  def extarctTiffMP(filename, dst_path, prefix, ext = 'png'):

    im = Image.open(filename)
    # compression = im.info["compression"]
    i = 0
    for page in ImageSequence.Iterator(im):
      np_img = np.array(page.convert("L"))
      dst_filename = os.path.join(dst_path, prefix + str(i) + '.' + ext)
      cv2.imwrite(dst_filename, np_img)
      i = i +1

      # result.append(np.fromarray(np.array(page)))


  @staticmethod
  def saveTiffMP(filename,images):
    imlist = []
    for m in images:
      imlist.append(Image.fromarray(m))

    imlist[0].save(filename, compression="group4", save_all=True,
                   append_images=imlist[1:])



  @staticmethod
  def saveTiffMPFromPath(dst_filename, src_path):
    src_files = FileUtility.getFolderImageFiles(src_path)

    imlist = []
    for i in tqdm(range(len(src_files)), ncols=100):
      src_file =  src_files[i]
      img = Image.open(src_file)
      img = img.convert('1')
      imlist.append(img)

    imlist[0].save(dst_filename, compression="group4", save_all=True,
                   append_images=imlist[1:])

  @staticmethod
  def saveImages(images,dst_path,prefix,counetr =0, ext='png'):
    c = counetr
    for i in tqdm(range(len(images)), ncols=100):
      image  = images[i]
      dst_filename = os.path.join(dst_path,prefix+str(c)+'.'+ext)
      c += 1
      cv2.imwrite(dst_filename,image)
    return c

  @staticmethod
  def mean_point(pnt1,pnt2):
    return [int( (pnt2[0] + pnt1[0])  /2),int((pnt2[1] + pnt1[1]) /2)]

  @staticmethod
  def points_distance(pnt1,pnt2):
     return ((pnt2[0] - pnt1[0]) ** 2 + (pnt2[1] - pnt1[1]) ** 2) ** 0.5

  @staticmethod
  def points_distance_x(pnt1,pnt2):
     return abs(pnt2[0] - pnt1[0])

  @staticmethod
  def points_distance_y(pnt1,pnt2):
     return abs(pnt2[1] - pnt1[1])


  @staticmethod
  def offset_point(pnt,offset):
    return [(pnt[0]+offset[0]),(pnt[1]+offset[1])]

  @staticmethod
  def rect_points_dim(rect):
    p1 = rect[0]
    p2 = rect[1]
    w = CvUtility.points_distance_x(p1,p2)
    h = CvUtility.points_distance_y(p1, p2)

    return [p1[0],p1[1],w,h]

  @staticmethod
  def rect_dim_points(rect):

    return [int(rect[0]),int(rect[1]) ,int(rect[0]+rect[2]),int(rect[1]+rect[3])]

  @staticmethod
  def add_border(image,border = None,color = (255,255,255)):
     if border:
       return cv2.copyMakeBorder(image, border[0], border[1], border[2], border[3], cv2.BORDER_CONSTANT, value=color)
     else: return image

  @staticmethod
  def remove_border(image,border):
      region = CvUtility.getImageRect(image)
      t,b,l,r = border
      x, y, w, h = region
      return image[y+t:y + h-b, x+l:x + w - r]
  
  @staticmethod
  def array2Image(arr):
    fit_image = np.squeeze(arr)
    return np.transpose(fit_image, (1, 2, 0))

  @staticmethod
  def image_to_tensor(image, target_shape):
    result = image.transpose((2, 0, 1))  # HWC->CHW
    result = result.reshape(target_shape)
    return result
  
  @staticmethod
  def getPointsBoundingBox(points):
    pnts = np.empty((0, 2), int)
    for pnt in points: 
        p = [np.array(pnt)]
        pnts = np.append(pnts, p, axis=0)

    x, y, w, h = cv2.boundingRect(pnts)

    return [x, y,  w,  h]
  
  @staticmethod
  def drawPoints(image,points,color=(0,255,0),size = 2):
    for point in points:
      cv2.circle(image,point,size,color,-1)




  @staticmethod
  def getNormAnchors(anchors):
    bb = getAnchorsBoundingBox(anchors)
    result = anchors
    for pnt in result:
      pnt[0] = bb[1]
      pnt[1] = bb[0]
    return result

  @staticmethod
  def filter_images_by_dim(src_path, min_width=-1,max_width=-1,min_height=-1,max_height=-1):
    if min_width == -1:
      min_w = 0
    if max_width == -1:
      max_w = sys.maxsize
    if min_height == -1:
      min_h = 0
    if max_height == -1:
      max_h = sys.maxsize

    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = []
    for src_file in src_files:
      img = cv2.imread(src_file)
      if img.shape[0] >= min_h and img.shape[0] <= max_h and img.shape[1] >=min_w and img.shape[1] <= max_w:
        dst_files.append(src_file)

    return dst_files

  @staticmethod
  def rotate_image(image,angle):

      result = imutils.rotate_bound(image, angle)

      return result

  def rotate_image2(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
      center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

  @staticmethod
  def rotate_image_batch(src_path,dst_path,angle):
    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files,dst_path,src_path)
    for i in tqdm(range(len(src_files)), ncols=100):
      src_image = cv2.imread(src_files[i])
      dst_image =  CvUtility.rotate_image(src_image,angle)
      cv2.imwrite(dst_files[i],dst_image)

  @staticmethod
  def rotate_image_batchs(src_path, dst_path, angles):
    src_files = FileUtility.getFolderImageFiles(src_path)
    dst_files = FileUtility.getDstFilenames2(src_files,  dst_path,src_path)
    for i in range(len(src_files)):
      src_image = cv2.imread(src_files[i])
      for angle in angles:
        dst_image = CvUtility.rotate_image(src_image, angle)
        dst_filename  = FileUtility.changeFilenamePostfix(dst_files[i],"_"+str(angle))
        cv2.imwrite(dst_filename, dst_image)

  @staticmethod
  def rotate_image_batch_(images_path,angle):
    src_files = FileUtility.getFolderImageFiles(src_path)

    for i in tqdm(range(len(src_files)), ncols=100):

      src_image = cv2.imread(src_files[i])
      dst_image =  CvUtility.rotate_image2(src_image,angle)
      cv2.imwrite(src_files[i], dst_image)

  @staticmethod
  def cvRect2Yolo(rect,size):
     r = (float(rect[0]) / size[1], float(rect[1]) / size[0],
          float(rect[2]) / size[1], float(rect[3]) / size[0])

     return (r[0] + r[2] / 2, r[1] + r[3] / 2, r[2], r[3])

  @staticmethod
  def rectInside(sub_rect,main_rect):
    return sub_rect[0] >= main_rect[0] and sub_rect[1] >= main_rect[1] and  sub_rect[2] <= main_rect[2] and  sub_rect[3] <= main_rect[3]

  @staticmethod
  def rect2CvRect(rct):
    return (rct[0],rct[1],rct[2] - rct[0] +1,rct[3] - rct[1] +1)

  @staticmethod
  def cvrect2Rect(cv_rct):
    return (cv_rct[0],cv_rct[1],cv_rct[0] + cv_rct[2] - 1,cv_rct[1] + cv_rct[3] - 1)

  @staticmethod
  def cvRectInside(sub_rect,main_rect):
    sub_rect_r = CvUtility.cvrect2Rect(sub_rect)
    main_rect_r = CvUtility.cvrect2Rect(main_rect)
    return CvUtility.rectInside(sub_rect_r, main_rect_r)



  @staticmethod
  def yolorect2Rect(yolo_rct,back_size = None):
    w2 = yolo_rct[2] / 2
    h2 = yolo_rct[3] / 2
    if back_size != None:
      result = ( (yolo_rct[0] - w2) * back_size[0], (yolo_rct[1] - h2) * back_size[1],
                 (yolo_rct[0] + w2) * back_size[0], (yolo_rct[1] + h2) * back_size[1])

    else : result = (yolo_rct[0] - w2, yolo_rct[1] - h2, yolo_rct[0] + w2, yolo_rct[1] + h2)
    return result
  @staticmethod
  def rect2Yolorect(rct,back_size):
      r = (rct[0] / back_size[0],rct[1] / back_size[1],rct[2] / back_size[0],rct[3] / back_size[1])
      w = r[2] - r[0]
      h = r[3] - r[1]
      return ( r[0] + w / 2,r[1]+ h / 2,w,h)


  @staticmethod
  def get_rect_size(rct):
    return (rct[2] - rct[0] +1 ,rct[3] - rct[1] +1 )

  @staticmethod
  def crop_rect(rct,crop_region):


    return ( rct[0] - crop_region[0], rct[1] - crop_region[1], rct[2] - crop_region[0], rct[3] - crop_region[1])


  @staticmethod
  def get_yolo_size(rct):
    return (rct[2],rct[3])

  @staticmethod
  def crop_yolo_region(yolo_rct,yolo_crop_region,back_size):

      size = CvUtility.get_yolo_size(yolo_crop_region)
      rct = CvUtility.yolorect2Rect(yolo_rct,back_size)
      crop_region = CvUtility.yolorect2Rect(yolo_crop_region,back_size)

      result = CvUtility.crop_rect(rct,crop_region)
      return CvUtility.rect2Yolorect(result,(size[0] * back_size[0],size[1] * back_size[1]))


  @staticmethod
  def yoloRectInside(sub_rect,main_rect):
    sub_rect_r = CvUtility.yolorect2Rect(sub_rect)
    main_rect_r = CvUtility.yolorect2Rect(main_rect)
    return CvUtility.rectInside(sub_rect_r,main_rect_r)

  @staticmethod
  def break_image(image, grid_cols, grid_rows, conflict_cols=0.0, conflict_rows=0.0):
    height, width, _ = image.shape
    cell_width = int(width / grid_cols)
    cell_height = int(height / grid_rows)

    offset_x = int(cell_width * conflict_cols / 2)
    offset_y = int(cell_height * conflict_rows / 2)

    result = []

    for i in range(grid_rows):
      pos_y = i * cell_height
      for j in range(grid_cols):
        pos_x = j * cell_width

        x1 = pos_x - offset_x
        if x1 < 0:
          x1 = 0
        x2 = pos_x + cell_width + offset_x
        if x2 >= width - 1:
          x2 = width - 1

        y1 = pos_y - offset_y
        if y1 < 0:
          y1 = 0
        y2 = pos_y + cell_height + offset_y
        if y2 >= height - 1:
          y2 = height - 1

        cur_patch = image[y1:y2, x1:x2]
        r = (x1, y1, x2, y2)
        result.append([cur_patch,r ])

    return result

  @staticmethod
  def get_contour_in_direct( contour, direct):
    result = contour[:]

    if len(contour) != 4:
      return []

    result = CvUtility.put_left_top_in_first(result)

    dist1 = np.linalg.norm(result[1] - result[0])
    dist2 = np.linalg.norm(result[2] - result[1])

    if (direct == 'horz' and dist1 < dist2) or (direct == 'vert' and dist1 > dist2):
      result = np.roll(result, -1, axis=0)

    return result

  @staticmethod
  def find_left_top_point( contour):
        leftmost = np.inf
        topmost = np.inf
        left_top_point = None

        for point in contour:
            if point[0][0] < leftmost or (point[0][0] == leftmost and point[0][1] < topmost):
                leftmost = point[0][0]
                topmost = point[0][1]
                left_top_point = (leftmost, topmost)

        return left_top_point

  @staticmethod
  def find_left_topest_point( contour):
        bb_lt = CvUtility.find_left_top_point(contour)
        min_dist = np.inf
        closest_point_index = 0

        for i, point in enumerate(contour):
            dist = np.sqrt((point[0][0] - bb_lt[0]) ** 2 + (point[0][1] - bb_lt[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_point_index = i

        return closest_point_index
  @staticmethod
  def put_left_top_in_first( contour):
    result = contour[:]
    lt_index = CvUtility.find_left_topest_point(contour)
    result = np.roll(result, -lt_index, axis=0)
    return result


  @staticmethod
  def rectify_rect_image( src_img, contour, dst_size):
    contour1 = CvUtility.get_contour_in_direct(contour, 'horz')
    contour = np.array(contour1, dtype=np.float32)

    dst = np.array([
      [0, 0],
      [dst_size[0], 0],
      [dst_size[0], dst_size[1]],
      [0, dst_size[1]]
    ], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(contour, dst)
    dst_img = cv2.warpPerspective(src_img, transform, (dst_size[0], dst_size[1]))

    return dst_img

  @staticmethod
  def get_rect_contour_dimension( approx_contour):
    contour = approx_contour.reshape(-1, 2)

    # Extract x and y coordinates
    x_coords = contour[:, 0]
    y_coords = contour[:, 1]

    # Calculate the bounding box dimensions
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    width = max_x - min_x
    height = max_y - min_y

    # Swap width and height if necessary
    if height > width:
      width, height = height, width

    return (width, height)

  @staticmethod
  def approximate_rect_contour( contour):
    epsilon = 0.1
    while True:
      approx = cv2.approxPolyDP(contour, epsilon, True)
      epsilon += 0.1
      if len(approx) <= 4:
        break

    if len(approx) != 4:
      None
    return approx

 
    

    
  
















