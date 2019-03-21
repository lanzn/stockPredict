#coding:utf-8
import DNN_moduel as dm

if __name__ == '__main__':
    #1.
    TARGET_PATH = "../../data/Train&Test/C1S4_newlabel_hybrid_timestep15_20180930/"  # 竖着拼四个
    TRAIN_FILE_NAME = "hybrid_train_set.csv"
    TRAIN_FILE = TARGET_PATH + TRAIN_FILE_NAME
    VALID_FILE_NAME = "hybrid_validate_set.csv"
    VALID_FILE = TARGET_PATH + VALID_FILE_NAME
    NEW_LABEL2_PATH = "../../data/Common/New_Label2/"

    final_shouyilv_1, precision_1, recall_1, f1_1, auc_1=dm.main(TRAIN_FILE,VALID_FILE,TARGET_PATH,NEW_LABEL2_PATH)


    #2.
    TARGET_PATH = "../../data/Train&Test/C1S4_newlabel_hybrid_timestep15_20180630/"  # 竖着拼四个
    TRAIN_FILE_NAME = "hybrid_train_set.csv"
    TRAIN_FILE = TARGET_PATH + TRAIN_FILE_NAME
    VALID_FILE_NAME = "hybrid_validate_set.csv"
    VALID_FILE = TARGET_PATH + VALID_FILE_NAME
    NEW_LABEL2_PATH = "../../data/Common/New_Label2/"

    final_shouyilv_2, precision_2, recall_2, f1_2, auc_2=dm.main(TRAIN_FILE,VALID_FILE,TARGET_PATH,NEW_LABEL2_PATH)

    #3.
    TARGET_PATH = "../../data/Train&Test/C1S4_newlabel_hybrid_timestep15_20180331/"  # 竖着拼四个
    TRAIN_FILE_NAME = "hybrid_train_set.csv"
    TRAIN_FILE = TARGET_PATH + TRAIN_FILE_NAME
    VALID_FILE_NAME = "hybrid_validate_set.csv"
    VALID_FILE = TARGET_PATH + VALID_FILE_NAME
    NEW_LABEL2_PATH = "../../data/Common/New_Label2/"

    final_shouyilv_3, precision_3, recall_3, f1_3, auc_3=dm.main(TRAIN_FILE,VALID_FILE,TARGET_PATH,NEW_LABEL2_PATH)

    #4.
    TARGET_PATH = "../../data/Train&Test/C1S4_newlabel_hybrid_timestep15_20171231/"  # 竖着拼四个
    TRAIN_FILE_NAME = "hybrid_train_set.csv"
    TRAIN_FILE = TARGET_PATH + TRAIN_FILE_NAME
    VALID_FILE_NAME = "hybrid_validate_set.csv"
    VALID_FILE = TARGET_PATH + VALID_FILE_NAME
    NEW_LABEL2_PATH = "../../data/Common/New_Label2/"

    final_shouyilv_4, precision_4, recall_4, f1_4, auc_4=dm.main(TRAIN_FILE,VALID_FILE,TARGET_PATH,NEW_LABEL2_PATH)



    #average
    final_shouyilv_aver=(final_shouyilv_1+final_shouyilv_2+final_shouyilv_3+final_shouyilv_4)/4
    precision_aver=(precision_1+precision_2+precision_3+precision_4)/4
    recall_aver = (recall_1 + recall_2 + recall_3 + recall_4) / 4
    f1_aver = (f1_1 + f1_2 + f1_3 + f1_4) / 4
    auc_aver = (auc_1 + auc_2 + auc_3 + auc_4) / 4

    #print
    print("4模型平均收益率：",final_shouyilv_aver)
    print("4模型平均precision:",precision_aver)
    print("4模型平均recall:", recall_aver)
    print("4模型平均f1:", f1_aver)
    print("4模型平均auc:", auc_aver)









