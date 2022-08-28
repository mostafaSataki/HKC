from enum import  Enum
import  numpy as np
import  os
from scipy.spatial import distance
from .DatasetUtility import *
from .FileUtility import *
import dlib
from  .FaceDetectionLandmark import  *

from  .FaceAlignment import *
from .FaceFeatureExtractor import  *
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, accuracy_score, balanced_accuracy_score
from .FaceDetection import *

from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats
from matplotlib import pyplot as plt
import datetime
from .FaceUtility import *
import json

class MatchData:
    def __init__(self,title,color, images_path, features_path, match_scores_path,
                     master_detection_type, slave_detection_type, master_landmark_type,slave_landmark_type,
                 alignment_type,    extractor_type,matcher_type=MatchType.EUCLIDEAN,border = None):
        self.title_ = title
        self.color_ = color
        self.images_path_ = images_path
        self.features_path_ = features_path
        self.match_scores_path_ = match_scores_path
        self.master_detection_type_ = master_detection_type
        self.slave_detection_type_ = slave_detection_type
        self.master_landmark_type_ = master_landmark_type
        self.slave_landmark_type_ = slave_landmark_type
        self.alignment_type_ = alignment_type
        self.extractor_type_ = extractor_type
        self.matcher_type_ =matcher_type
        self.MatchType_ = MatchType
        self.border_ = border

class FaceRecognition:
    def __init__(self,detections_models=None,landmark_models=None,alignment_models=None,extractor_models=None):

        self._detector_landmark_ = FaceDetectionLandmark(detections_models,landmark_models)

        self._alignment = FaceAlignment(alignment_models)
        self._extractor = FaceFeatureExtrator(extractor_models)

    def draw_landmark(self, image, face_region, landmark):
        for i in range(0, len(landmark)):
            cv2.circle(image, (landmark[i][0], landmark[i][1]), 3, (0, 0, 255), -1)
        cv2.rectangle(image, (face_region[0], face_region[1]), (face_region[0] + face_region[2], face_region[1] + face_region[3]), (0, 255,0), 2)
        return image
    def show_landmark(self,image,face_region,landmark):
        image2 = image.copy()
        draw_image = self.draw_landmark(image2,face_region,landmark)
        cv2.imshow("landmark",draw_image)
        cv2.waitKey(0)
    def save_region_landmark_asjson(self, region, landmark, filename):
        with open(filename, 'w') as f:
            json.dump({"region": region, "landmark": landmark}, f)

    def load_region_landmark_asjson(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            region = data["region"]
            landmark = data["landmark"]
        return region,landmark

    def feature_extract(self,bgr_image,rl_filename =None):
        rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
        if not(rl_filename is None) and  os.path.exists(rl_filename):
            face_region, face_landmark = self.load_region_landmark_asjson(rl_filename)
        else:
            face_region, face_landmark = self._detector_landmark_.extract(bgr_image, rgb_image,
                                                                          self._master_detection_type,
                                                                          self._slave_detection_type,
                                                                          self._master_landmark_type,
                                                                          self._slave_landmark_type)
            if not(rl_filename is None):
                self.save_region_landmark_asjson(face_region,face_landmark,rl_filename)





        # self.show_landmark(bgr_image,face_region,face_landmark)

        rgb_face_chip = self._alignment.extract(bgr_image, rgb_image, face_region, face_landmark, self._alignment_type)
        rgb_face_chip,face_chip  = self.normalize_model_input(rgb_face_chip)


        face_feature,face_chip_processed = self._extractor.feature_extract(face_chip, rgb_face_chip, self._extractor_type)
        return face_feature,face_chip, face_chip_processed

    def _get_model_input_size(self):
        result = (224,224)
        if self._extractor_type == FaceFeatureType.DLIB:
            result = (150,150)
        elif self._extractor_type == FaceFeatureType.IR50:
            result = (112, 112)
        elif self._extractor_type == FaceFeatureType.FACENET_128:
            result = (160, 160)
        elif self._extractor_type == FaceFeatureType.FACENET_512:
            result = (160, 160)
        elif self._extractor_type == FaceFeatureType.OPENVINO:
            result = (128,128)
        return result

    def normalize_model_input(self,bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self._extractor_type == FaceFeatureType.SF:
            return bgr_image,rgb_image
        else :
            image_size = CvUtility.getImageSize(bgr_image)
            model_input_size = self._get_model_input_size()
            if image_size != model_input_size:
                bgr_image = cv2.resize(bgr_image,model_input_size)
                rgb_image = cv2.resize(rgb_image, model_input_size)

            if self._extractor_type == FaceFeatureType.FACENET_128 or self._extractor_type == FaceFeatureType.FACENET_512:
                bgr_image = bgr_image.astype(np.float)
                rgb_image = rgb_image.astype(np.float)

                bgr_image /= 255
                rgb_image /= 255

                bgr_image = np.expand_dims(bgr_image, axis=0)
                rgb_image = np.expand_dims(rgb_image, axis=0)

            return bgr_image,rgb_image

    def _feature_extract(self,image,filename = None):
        bgr_image = image
        if self._border:
            white = [255, 255, 255]
            bgr_image = cv2.copyMakeBorder(bgr_image, self._border[0], self._border[1], self._border[2], self._border[3],
                                       cv2.BORDER_CONSTANT, value=white)

        if self._master_detection_type == DetectionType.CV_YUNET:
            bgr_image,_ = CvUtility.fitOnSizeMat(bgr_image,(150,150))

        rgb_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
        face_region,face_landmark = self._detector_landmark_.extract(bgr_image,rgb_image,self._master_detection_type,
                                                                     self._slave_detection_type,self._master_landmark_type,self._slave_landmark_type)

        if self._draw_path :
            if not os.path.exists(self._draw_path):
                os.makedirs(self._draw_path)

            view = bgr_image.copy()
            view = FaceUtility.draw_region_landmark(view,face_region,face_landmark)
            # cv2.imshow("view",view)
            # cv2.waitKey(0)
            fname = FileUtility.getFilename(filename)
            cv2.imwrite(os.path.join(self._draw_path ,fname),view)


        face_chip = self._alignment.extract(bgr_image,rgb_image,face_region, face_landmark,self._alignment_type)
        face_chip,rgb_face_chip = self.normalize_model_input(face_chip)





        face_feature,_ = self._extractor.feature_extract(face_chip,rgb_face_chip,self._extractor_type)
        return face_feature,rgb_face_chip

    def save_feature(self, feature, filename):
        np.save(filename, feature)

    def load_feature(self, filename):
        try:
            result = np.load(filename)
        except:
            result = None
            print(filename)
        return result

    def save_features(self):

        for src_path,dst_path in self._images_map.items():
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            src_filenames = FileUtility.getFolderImageFiles(src_path)
            dst_chip_filenames = FileUtility.getDstFilenames2(src_filenames,src_path, dst_path)
            dst_feature_filenames = FileUtility.changeFilesExt(dst_chip_filenames, "npy")

            sum = 0
            for j in tqdm(range(len(src_filenames)), ncols=100):

                if not os.path.exists(dst_feature_filenames[j]):
                    image = cv2.imread(src_filenames[j], 1)
                    print(src_filenames[j])
                    start = datetime.datetime.now()
                    feature,rgb_face_chip = self._feature_extract(image,src_filenames[j])
                    end = datetime.datetime.now()
                    c = end - start
                    t = int(c.total_seconds() * 1000)
                    # sum += t
                    # print("time:", t)
                    cv2.imwrite(dst_chip_filenames[j], rgb_face_chip)
                    self.save_feature(feature, dst_feature_filenames[j])
            print("time total:",sum / len(src_filenames))

    def load_features_from_files(self, filenames):
        features = []
        for i in tqdm(range(len(filenames)), ncols=100):
            filename = filenames[i]
            features.append(self.load_feature(filename))
        return features

    def load_features_from_path(self):
        all_src_image_files = []
        all_feature_filenames = []
        for src_path,dst_path in self._images_map.items():
            dst_chip_filenames = FileUtility.getFolderImageFiles(dst_path)
            feature_filenames = FileUtility.changeFilesExt(dst_chip_filenames, "npy")
            src_image_files = FileUtility.getDstFilenames2(dst_chip_filenames,dst_path,src_path)

            all_src_image_files.extend(src_image_files)
            all_feature_filenames.extend(feature_filenames)

        all_features = self.load_features_from_files(all_feature_filenames)

        return all_features, all_src_image_files

    def extract_features_from_path(self):
        self.save_features()
        return self.load_features_from_path()

    def create_face_detectors(self):
        models_path = r'E:\Database\data_deep\face_models'
        landmark_filename = os.path.join(models_path, 'dlib\shape_predictor_5_face_landmarks.dat')
        face_rec_model_filename = os.path.join(models_path, r'dlib\dlib_face_recognition_resnet_model_v1.dat')
        cascade_face_filename = r'I:\Library\opencv4.5.3\opencv\data\haarcascades\haarcascade_frontalface_default.xml'

        self._dlib_detector = dlib.get_frontal_face_detector()
        self._cascade_detector = cv2.CascadeClassifier(cascade_face_filename)
        self._sp = dlib.shape_predictor(landmark_filename)

    def _face_detect(self, image):
        dets = self._dlib_detector(image, 1)
        if len(dets) == 0:
            cv_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = self._cascade_detector.detectMultiScale(cv_img, 1.1, 5, minSize=(30, 30),
                                                            flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                dets.append(dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
        return dets

    def _get_det_area(self, det):
        return (det.right() - det.left()) * (det.bottom() - det.top())

    def _get_max_det(self, dets):
        if len(dets) == 0:
            return None
        max_det_area = 0
        max_index = -1
        for i, det in enumerate(dets):
            cur_area = self._get_det_area(det)
            if cur_area > max_det_area:
                max_det_area = cur_area
                max_index = i

        return dets[max_index]

    def face_detect(self,image,master_type = DetectionType.DLIB,slave_type = DetectionType.CV_CASCADE):
        if master_type == DetectionType.DLIB:
           dlib_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           dets = self._face_detect(dlib_img)
           if len(dets) == 0:
               cv_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
               faces = self._cascade_detector.detectMultiScale(cv_img, 1.1, 5, minSize=(30, 30),
                                                               flags=cv2.CASCADE_SCALE_IMAGE)
               for (x, y, w, h) in faces:
                   dets.append(dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))

        max_det = self._get_max_det(dets)
        if max_det == None:
            return None
        shape = self._sp(dlib_img, max_det)
        # face_feature = self._facerec.compute_face_descriptor(dlib_img, shape)
        face_chip = dlib.get_face_chip(dlib_img, shape)
        face_feature = self._face_rec.compute_face_descriptor(face_chip)
        return face_feature

    def extract_genuine_imposter_scores(self, genuine, imposter, features, match_type = MatchType.EUCLIDEAN):
        genuine_scores = []
        imposter_scores = []

        for i in tqdm(range(len(genuine)), ncols=100):
            gen = genuine[i]
            x1, x2 = gen
            genuine_scores.append(self.get_distance(features[x1], features[x2],match_type))

        for i in tqdm(range(len(imposter)), ncols=100):
            imp = imposter[i]
            x1, x2 = imp
            imposter_scores.append(self.get_distance(features[x1], features[x2],match_type))

        return np.array(genuine_scores), np.array(imposter_scores)

    def cross_match_scores(self, features, match_type=MatchType.EUCLIDEAN):
        scores = []
        pairs = []
        for i in tqdm(range(len(features)-1), ncols=100):
            for j in range(len(features)):
                scores.append(self.get_distance(features[i], features[j], match_type))
                pairs.append([i,j])

        return np.array(scores),np.array(pairs)

    def brute_force_scores(self, gallery_features,prob_features, match_type=MatchType.EUCLIDEAN):
        scores = []

        for i in tqdm(range(len(prob_features)), ncols=100):
            score_row = []
            for j in range(len(gallery_features)):
                score_row.append(self.get_distance(prob_features[i], gallery_features[j], match_type))
            scores.append(score_row)

        return np.array(scores)


    def euclidean_distance(self, a, b):
       return distance.euclidean(a, b)

    def cosine_distance(self, a, b):
       return distance.cosine(a, b)

    def euclidean_distance_np(a, b):
        if type(a) == list:
            a = np.array(a)

        if type(b) == list:
            b = np.array(b)

        euclidean_distance = a - b
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def get_distance(self,a,b,type = MatchType.EUCLIDEAN):
        if type == MatchType.EUCLIDEAN:
            return  self.euclidean_distance(a,b)
        elif type == MatchType.COSINE:
            return  self.cosine_distance(a,b)

    def load_genuine_imposter_scores(self,title, images_path, features_path, match_scores_path,
        master_detection_type,slave_detection_type,master_landmark_type,slave_landmark_type,
                                     alignment_type,extractor_type,matcher_type =MatchType.EUCLIDEAN,border = None
                                     ):

        self._title = title
        self._master_detection_type = master_detection_type
        self._slave_detection_type = slave_detection_type
        self.master_landmark_type = master_landmark_type
        self.slave_landmark_type = slave_landmark_type
        self._alignment_type = alignment_type
        self._extractor_type = extractor_type
        self._border = border

        genuine_mc_filename = os.path.join(match_scores_path, 'genuine.npy')
        imposter_mc_filename = os.path.join(match_scores_path, 'imposter.npy')

        if (not (os.path.exists(genuine_mc_filename) and (os.path.exists(imposter_mc_filename)))):
            features, files = self.extract_features_from_path(images_path, features_path)
            genuine, imposter = DatasetUtility.extract_imposter_genuine_from_files(files, '_')
            genuine_scores,imposter_scores = self.extract_genuine_imposter_scores(genuine,imposter,features,matcher_type)

            np.save(genuine_mc_filename,genuine_scores)
            np.save(imposter_mc_filename, imposter_scores)

        else:
            genuine_scores = np.load(genuine_mc_filename)
            imposter_scores = np.load(imposter_mc_filename)

        return genuine_scores, imposter_scores


    def load_genuine_imposter_scores(self,title, images_path, features_path, match_scores_path,
        master_detection_type,slave_detection_type,master_landmark_type,slave_landmark_type,alignment_type,
                                     extractor_type,matcher_type =MatchType.EUCLIDEAN,border = None
                                     ):
        self._title = title
        self._master_detection_type = master_detection_type
        self._slave_detection_type = slave_detection_type
        self._master_landmark_type = master_landmark_type
        self._slave_landmark_type = slave_landmark_type

        self._alignment_type = alignment_type
        self._extractor_type = extractor_type
        self._border = border

        genuine_mc_filename = os.path.join(match_scores_path, 'genuine.npy')
        imposter_mc_filename = os.path.join(match_scores_path, 'imposter.npy')

        if (not (os.path.exists(genuine_mc_filename) and (os.path.exists(imposter_mc_filename)))):
            features, feature_files,_ = self.extract_features_from_path(images_path, features_path)
            genuine, imposter = DatasetUtility.extract_imposter_genuine_from_files(feature_files, '_')
            genuine_scores,imposter_scores = self.extract_genuine_imposter_scores(genuine,imposter,features,matcher_type)

            np.save(genuine_mc_filename,genuine_scores)
            np.save(imposter_mc_filename, imposter_scores)

        else:
            genuine_scores = np.load(genuine_mc_filename)
            imposter_scores = np.load(imposter_mc_filename)

        return genuine_scores, imposter_scores

    def get_accuracy(self, images_path, features_path, match_scores_path,
                                 master_detection_type, slave_detection_type, master_landmark_type,slave_landmark_type, alignment_type,
                                 extractor_type,fpr_thresh, matcher_type=MatchType.EUCLIDEAN):
        genuine_scores, imposter_scores = self.load_genuine_imposter_scores(images_path, features_path,
                                                                                match_scores_path, master_detection_type,
                                                                                slave_detection_type,
                                                                                matser_landmark_type,
                                                                                slave_landmark_type, alignment_type,
                                                                                extractor_type,
                                                                                matcher_type)


        labels = [0] * len(genuine_scores) + [1] * len(imposter_scores)
        scores = np.concatenate([genuine_scores, imposter_scores])
        fpr, tpr, thresholds = roc_curve(np.array(labels), scores, drop_intermediate=False)
        # scores2 = np.where(scores > 0.5771760674040008, 1, 0)
        # value = accuracy_score(np.array(labels), scores2)
        # print(value)



        idx = 0
        all_thresh = False
        if all_thresh:
            with open(r'd:\result.txt','w') as f:
                for x in fpr:
                    str1 ="fpr:"+str(fpr[idx])+" tpr:"+str(tpr[idx])+" threshold:"+str(thresholds[idx])+"\n"
                    f.write(str1)
                    # if x > fpr_thresh:
                    #      return tpr[idx],thresholds[idx]
                    idx += 1
                f.close()
        else :
                for x in fpr:
                    if x > fpr_thresh:
                         return tpr[idx],thresholds[idx]
                    idx += 1

    def draw_DET_curve(self, fnmr, fmr, output_path, algo_name, det_resolution=100000):

        x_ax = []
        size = fnmr.shape[0]
        if size > det_resolution:
            num = 0
            for i in range(det_resolution):
                x_ax.append(math.ceil(num))
                num = num + size / det_resolution
        else:
            x_ax = range(0, size)

        fmr_o = np.zeros([1, len(x_ax)])
        fnmr_o = np.zeros([1, len(x_ax)])
        for j in range(size):
            fmr_o[0, j] = fmr[ int(x_ax[j])]
            fnmr_o[0, j] = fnmr[ int(x_ax[j])]

        plt.figure(2)
        plt.semilogx(fmr_o.tolist()[0], fnmr_o.tolist()[0], color="blue")
        tt = np.logspace(-5, 0, 1000)
        plt.semilogx(tt, tt, color=[0.7, 0.7, 0.7])
        plt.xlabel('FMR')
        plt.ylabel('FNMR')
        plt.title('DET curve')
        plt.savefig(os.path.join(output_path , algo_name + '_DET_curve.png'))

        return

    def save_score_to_file(self,filename, scores,thresholds):
        with open(filename,'w') as f :
            for i,t in enumerate(thresholds):
                # f.write('%d %1.6f %1.6 \n' % (i,t,scores[i]))
                f.write("{} {} {} \n".format(i,t,scores[i]))


    def get_fnmr(self, genuine_scores):
        GS = np.sort(genuine_scores)
        GS = np.flip(GS)

        thresholds = np.unique(GS)
        thresholds = np.flip(thresholds)

        g_len = GS.size
        g_len_f = float(g_len)

        fnmr = np.zeros_like(thresholds)

        i = 0
        index = 0
        for t in thresholds:
            if i >= g_len:
                break

            while i < g_len and GS[i] >= t:
                i += 1


            fnmr[index] = i / g_len_f
            index += 1

        return fnmr,thresholds



    def get_fmr(self, imposter_scores, thresholds):
        i_len = imposter_scores.size
        i_len_f = float(i_len)
        fmr = np.zeros_like(thresholds)
        IS = np.sort(imposter_scores)
        GT = np.flip(thresholds)

        i=0
        for index,t in enumerate( GT):
            if i >= i_len:
                break
            while i < i_len and IS[i] < t:
                i += 1
            fmr[index] = i / i_len_f

        return fmr


    def get_fnmr_fmr(self, genuine_scores,imposter_scores):
        fnmr,thresholds = self.get_fnmr(genuine_scores)
        fmr = self.get_fmr(imposter_scores,thresholds)
        fnmr = np.flip(fnmr)
        return fnmr,fmr



    def get_det(self, genuine_scores,imposter_scores):
        IS = np.sort(imposter_scores)

        i_len = imposter_scores.size
        i_len_f = float(i_len)
        fmr = np.zeros([genuine_thresholds.size])
        IS = np.sort(imposter_scores)
        GT = genuine_thresholds[::-1]
        fnmr = fnmr[::-1]
        i = 0
        for index, t in enumerate(GT):
            if i >= i_len:
                break
            while i < i_len and IS[i] < t:
                i += 1
            fmr[index] = i / i_len_f
            print(index, i, t, fnmr[index], fmr[index])

        return fmr, fnmr

    def get_accuracy2(self,mds,result_path):

        fig = plt.figure(figsize=(8, 6))

        plt.title("DET")
        plt.xlabel("FMR")
        plt.ylabel("FMNR")


        for md in mds:
            genuine_scores, imposter_scores = self.load_genuine_imposter_scores(md.title_, md.images_path_, md.features_path_,
                                                                                md.match_scores_path_, md.master_detection_type_,
                                                                                md.slave_detection_type_,
                                                                                md.master_landmark_type_,
                                                                                md.slave_landmark_type_,
                                                                                md.alignment_type_,
                                                                                md.extractor_type_,
                                                                                md.matcher_type_)


            fnmr,fmr = self.get_fnmr_fmr(genuine_scores, imposter_scores)
            plt.plot(fmr, fnmr,color =md.color_,label=md.title_)

        # plt.plot(fmr_b, fnmr_b,color = 'b',label="MatIRAN2")
        plt.xlim(10**-5, 10**0)
        plt.ylim(0.0, 0.1)

        plt.xscale("log")
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.legend(loc=0)
        plt.savefig(os.path.join(result_path, 'DET.png'))
        plt.show()




        # plt.savefig(os.path.join( result_path , 'DET-ver.png'))

        # for i, v in enumerate(fnmr):
        #   print(i,v,fmr[i], thresholds1[i])

        # for t in thresholds1:
        # for i,t in enumerate(thresholds2):
        #    if i > 1000:
        #      break
        #    print(-i,fmr[-i],thresholds2[-i])
        # size = fmr.shape[0]
        # fnmr_o = np.zeros([size])
        # # fnmr_o = np.zeros([1, len(x_ax)])
        # for j in range(size):
        #     if
        #     fmr_o[j] = fmr[int(x_ax[j])]
        #     fnmr_o[0, j] = fnmr[int(x_ax[j])]
        #
        # plt.title("Matplotlib demo")
        # plt.xlabel("x axis caption")
        # # plt.ylabel("y axis caption")
        # plt.plot(fmr)
        # plt.show()
        # plt.savefig(os.path.join( result_path , 'DET-ver.png'))
        #


        #
        # plt.savefig(self.path + 'DET-ver.pdf', bbox_inches='tight')
        # plt.clf()
        #
        #
        # plt.show()

        # g_a_min = np.min(genuine_scores_a)
        # g_a_max = np.max(genuine_scores_a)
        # i_a_min = np.min(imposter_scores_a)
        # i_a_max = np.max(imposter_scores_a)
        # print("g_a_min:",g_a_min)
        # print("g_a_max:", g_a_max)
        # print("i_a_min:", i_a_min)
        # print("i_a_max:", i_a_max)
        #
        # genuine_scores_b, imposter_scores_b = self.load_genuine_imposter_scores(md_b.images_path_, md_b.features_path_,
        #                                                                     md_b.match_scores_path_, md_b.master_detection_type_,
        #                                                                     md_b.slave_detection_type_,
        #                                                                     md_b.landmark_type_, md_b.alignment_type_,
        #                                                                     md_b.extractor_type_,
        #                                                                     md_b.matcher_type_)
        #
        #
        # g_b_min = np.min(genuine_scores_b)
        # g_b_max = np.max(genuine_scores_b)
        # i_b_min = np.min(imposter_scores_b)
        # i_b_max = np.max(imposter_scores_b)
        # print("g_b_min:",g_b_min)
        # print("g_b_max:", g_b_max)
        # print("i_b_min:", i_b_min)
        # print("i_b_max:", i_b_max)
        #
        # genuine_scores_b = genuine_scores_b * 0.69634938950595008717669920845425
        # imposter_scores_b = imposter_scores_b * 0.69634938950595008717669920845425
        #
        # # Calculating stats for classifier A
        # stats_a = get_eer_stats(genuine_scores_a, imposter_scores_a)
        #
        # # Calculating stats for classifier B
        # stats_b = get_eer_stats(genuine_scores_b, imposter_scores_b)
        #
        #
        # # Generating CSV report
        # generate_eer_report([stats_a, stats_b], ['A', 'B'],os.path.join(result_path, 'pyeer_report.csv'))
        #
        # # Generating HTML report
        # generate_eer_report([stats_a, stats_b], ['A', 'B'],os.path.join(result_path, 'pyeer_report.html'))
        #
        # # Generating Latex report
        # generate_eer_report([stats_a, stats_b], ['A', 'B'], os.path.join(result_path,'pyeer_report.tex'))
        #
        # # Generating JSON report
        # generate_eer_report([stats_a, stats_b], ['A', 'B'], os.path.join(result_path,'pyeer_report.json'))
        #
        # # Exporting error rates (Exporting FMR and FNMR to a CSV file)
        # # This is the DET curve, the ROC curve is a plot of FMR against 1 - FNMR
        # export_error_rates(stats_a.fmr, stats_a.fnmr, os.path.join(result_path,'A_DET.csv'))
        # export_error_rates(stats_b.fmr, stats_b.fnmr,os.path.join(result_path, 'B_DET.csv'))
        #
        # # Plotting
        # plot_eer_stats([stats_a, stats_b], ['A', 'B'])



    def load_cross_match_scores(self, images_path, features_path, match_result_path,
                                 master_detection_type, slave_detection_type, landmark_type, alignment_type,
                                 extractor_type, matcher_type=MatchType.EUCLIDEAN):


        self._master_detection_type = master_detection_type
        self._slave_detection_type = slave_detection_type
        self._landmark_type = landmark_type
        self._alignment_type = alignment_type
        self._extractor_type = extractor_type

        ms_filename = os.path.join(match_scores_path, 'scores.npy')
        indexs_filename = os.path.join(match_scores_path, 'indexs.npy')

        if (not (os.path.exists(ms_filename)) and not (os.path.exists(indexs_filename))):
            features, files = self.extract_features_from_path(images_path, features_path)
            scores,indexs = self.cross_match_scores(features,features,matcher_type)

            np.save(ms_filename,scores)
            np.save(indexs_filename, indexs)

        else:
            scores = np.load(ms_filename)
            indexs = np.load(indexs_filename)

        return scores, indexs


    def cross_match(self, images_path, features_path, match_scores_path,match_result_path,
                                 master_detection_type, slave_detection_type, landmark_type, alignment_type,
                                 extractor_type,threshold, matcher_type=MatchType.EUCLIDEAN):

        scores, indexs = self.load_cross_match_scores(images_path, features_path,
                                                                                match_scores_path, master_detection_type,
                                                                                slave_detection_type,
                                                                                landmark_type, alignment_type,
                                                                                extractor_type,
                                                                                matcher_type)

        # for in range(len())
        # idx = 0
        # for x in fpr:
        #     if x > fpr_thresh:
        #          return tpr[idx],thresholds[idx]
        #     idx += 1


    def brute_force_match(self, gallery_features, gallery_image_files,
                          prob_features, prob_image_files, threshold, match_result_path,
                          matcher_type=MatchType.EUCLIDEAN,Rank = 1):
        if (len(gallery_image_files) == 0 or len(prob_image_files) == 0):
            return


        FileUtility.createClearFolder(match_result_path)

        scores = self.brute_force_scores(gallery_features,prob_features,matcher_type)

        true_count = 0
        for i in tqdm(range(len(scores)), ncols=100):

            prob_image_file = prob_image_files[i]

            dst_branch = FileUtility.copy_file_to_folder(prob_image_file,match_result_path,'',True)
            match_scores = scores[i]
            min_thresh = 100
            gallery_image_file = ''
            if Rank == 1:
                for j in range(len(match_scores)):
                    if match_scores[j] < threshold and match_scores[j] < min_thresh:


                        gallery_image_file = gallery_image_files[j]

                        min_thresh = match_scores[j]
                        # FileUtility.copy_file_to_folder(gallery_image_file, dst_branch, "_" + str(match_scores[j]))

                if gallery_image_file != '':
                    prob_fname = FileUtility.getFileTokens(prob_image_file)[1]
                    if gallery_image_file != '':
                        gallery_fname = FileUtility.getFileTokens(gallery_image_file)[1]
                        if prob_fname == gallery_fname :
                            true_count += 1
                        else:
                             print('error:',prob_fname)

                    FileUtility.copy_file_to_folder(gallery_image_file, dst_branch, "_gallery")
                    print(min_thresh)


            else :
                prob_fname = FileUtility.getFileTokens(prob_image_file)[1]
                ms = np.array(match_scores)
                gi = np.array(gallery_image_files)
                idx = np.argsort(ms)
                ms2 = (np.array(ms)[idx]).tolist()[:Rank]
                gi2 = (np.array(gi)[idx]).tolist()[:Rank]

                dst_branch = FileUtility.copy_file_to_folder(prob_image_file, match_result_path, '', True)
                flag = 0
                for j in range(Rank):
                    g_filename = gi2[j]
                    gallery_fname = FileUtility.getFileTokens(g_filename)[1]
                    if prob_fname == gallery_fname:
                        flag = 1
                    FileUtility.copy_file_to_folder(gi2[j], dst_branch, "_gallery")
                if flag == 0:
                    print(prob_fname)
                true_count += flag

                # for j in range(len(match_scores)):
                #
                #     if match_scores[j] < threshold :
                #         # gallery_feature_file = gallery_feature_files[j]
                #         gallery_image_file = gallery_image_files[j]
                #
                #
                #         min_thresh = match_scores[j]
                #         FileUtility.copy_file_to_folder(gallery_image_file, dst_branch, "_"+str(match_scores[j]))
                #
        print("acc:",float(true_count) / len(scores)*100)

    def _get_org_name(self, value):
        tokens = value.split('_')
        if len(tokens) >  1:
            tokens.pop()
            if len(tokens) > 1:
                return '_'.join(tokens)
            else:
                return tokens[0]
        else:
            return value

    def _get_new_map_paths(self,old_map,new_paths,dst_path):
        new_names = {}

        names = {}
        for key,value in old_map.items():
            new_names[key] = value
            name = FileUtility.getFolderLabel(value)

            org_name = self._get_org_name(name)
            if not(org_name in names):
                names[org_name] = 1
            else:
                names[org_name] += 1


        for new_path in new_paths:
          if not(new_path in new_names):
            name = FileUtility.getFolderLabel(new_path)
            new_name = name
            if name in names:
                count = names[name] +1
                new_name = name + '_' + str(count)
                names[name] = count
            else:
                new_name = name + '_1'
                names[name] = 1
            new_names[new_path] = new_name

        for key,value in new_names.items():
            new_names[key] = os.path.join(dst_path,value)
        return new_names


    def _write_config_file(self,config_filename):
        data = {}
        data['master_detection_type'] = self._master_detection_type.value
        data['slave_detection_type'] = self._slave_detection_type.value
        data['master_landmark_type'] = self._master_landmark_type.value
        data['slave_landmark_type'] = self._slave_landmark_type.value
        data['alignment_type'] = self._alignment_type.value
        data['extractor_type'] = self._extractor_type.value

        data['all_map_paths'] = self._all_map_path


        with open(config_filename, 'w') as outfile:
            json.dump(data, outfile)


    def _read_config_file(self,config_filename):
        try:
            with open(config_filename, 'r') as f:
                data = json.load(f)
                return data
        except:
            os.remove(config_filename)
            return None

    def _compare_config(self,filename):
        with open(filename, 'r') as f:
            data = json.load(f)

            if data['master_detection_type'][0] != self._master_detection_type.value[0]:
                return False
            if data['slave_detection_type'][0] != self._slave_detection_type.value[0]:
                return False
            if data['master_landmark_type'][0] != self._master_landmark_type.value[0]:
                return False
            if data['slave_landmark_type'][0] != self._slave_landmark_type.value[0]:
                return False
            if data['alignment_type'][0] != self._alignment_type.value[0]:
                return False
            if data['extractor_type'][0] != self._extractor_type.value[0]:
                return False

            return True

    def _get_images_map(self,images_path):
        result = {}
        for image_path in images_path:
            result[image_path] = self._all_map_path[image_path]
        return result
    def init(self,master_detection_type, slave_detection_type,
                                        master_landmark_type, slave_landmark_type,
                                        alignment_type, extractor_type):
        self._master_detection_type = master_detection_type
        self._slave_detection_type = slave_detection_type
        self._master_landmark_type = master_landmark_type
        self._slave_landmark_type = slave_landmark_type
        self._alignment_type = alignment_type
        self._extractor_type = extractor_type
        
    def extract_features_from_path_proc(self, images_path, session_path, master_detection_type, slave_detection_type,
                                        master_landmark_type, slave_landmark_type,
                                        alignment_type, extractor_type, border =None, draw_path = None):
        self._master_detection_type = master_detection_type
        self._slave_detection_type = slave_detection_type
        self._master_landmark_type = master_landmark_type
        self._slave_landmark_type = slave_landmark_type
        self._alignment_type = alignment_type
        self._extractor_type = extractor_type
        self._border = border
        self._session_path = session_path
        self._draw_path = draw_path

        if not os.path.exists(session_path):
            os.makedirs(session_path)
        if draw_path and not os.path.exists(draw_path):
            os.makedirs(draw_path)

        self._config_filename = os.path.join(session_path, 'config.json')
        self._all_map_path = {}
        if os.path.exists(self._config_filename):
            if not self._compare_config(self._config_filename):
               print("session config  is different")
               return
            else:
                config_data = self._read_config_file(self._config_filename)
                if config_data:
                  self._all_map_path = config_data['all_map_paths']



        self._all_map_path = self._get_new_map_paths(self._all_map_path,images_path,self._session_path )
        self._write_config_file(self._config_filename)


        self._images_map = self._get_images_map(images_path)

        return self.extract_features_from_path()






