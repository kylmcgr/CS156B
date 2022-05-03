from preprocess_utils import (
    gen_cnn_basic,
    gen_cnn_densenet,
    gen_cnn_resnet,
    imputation_test,
)

OUTPUT_PATH = "/home/mcgee/CS156b/preprocess/"


cnn_basic = gen_cnn_basic()
cnn_resnet = gen_cnn_resnet()
cnn_densenet = gen_cnn_densenet()

# run_model(cnn_basic, OUTPUT_PATH + "cnn_basic.csv")
# run_model(cnn_resnet, OUTPUT_PATH + "cnn_resnet.csv")
imputation_test(cnn_basic, OUTPUT_PATH + "cnn_basic_impute")
