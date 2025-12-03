In /RIG root directory

python AdaFace\inference\dataset_preprocessor.py --input_dir dataset\classroom
To process all images and detect faces

python AdaFace\inference\segment_dataset.py
to segment dataset based on criteria


python face_recognition_live.py --session_name test_logging1

python face_recognition_server.py --gallery gallery\adaface_ir101.pkl --host 0.0.0.0 --port 5000
python face_recognition_client.py --server http://localhost:5000 --session_name client_server_testing_1