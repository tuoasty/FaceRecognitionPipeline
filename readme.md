In /RIG root directory

python AdaFace\inference\dataset_preprocessor.py --input_dir dataset\classroom
To process all images and detect faces

python AdaFace\inference\segment_dataset.py --input_dir output\v1\probe_labeled\positive --metadata_file output\v1\probe_labeled\probe_positive_metadata.json --output_dir output\v1\probe_labeled\segmented
to segment dataset based on criteria

To generate gallery
python AdaFace\inference\enroll_students.py --enrollment_dir dataset\enrollment --gallery_path AdaFace\inference\gallery\adaface_ir101_2.pkl --architecture ir_101 --model_type adaface

To match
python AdaFace\inference\face_matcher.py --single_image AdaFace\inference\samples\random\random_6.jpg --gallery_path AdaFace\inference\gallery\adaface_ir101_2.pkl --model_type adaface --architecture ir_101

To label
python AdaFace\inference\probe_labeler.py --probe_dir output\preprocessed\probe_positive --output_dir output\preprocessed\probe_positive_labeled --metadata output\preprocessed\probe_positive_metadata.json --gallery_path AdaFace\inference\gallery\adaface_ir101_3.pkl --model_type adaface --architecture ir_101

In /inference directory
python face_recognition_live.py --session_name test_logging1

python face_recognition_server.py --gallery gallery\adaface_ir101.pkl --host 0.0.0.0 --port 5000
python face_recognition_client.py --server http://10.22.78.202:5000 --session_name remote_test_1

http://10.22.78.202:5000

python AdaFace\inference\embedding_generator.py --dataset_root dataset --output_root output\v0 --model_type adaface --architecture ir_101