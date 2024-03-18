## script to lunch the experimentations :
BASE=$(pwd)
echo "---------------------------------------"
echo "Process the data"
python3 dataset.py 	--test_content_dir $BASE'/data/content'\
			--test_style_dir $BASE'/data/style'
echo "---------------------------------------"
echo "Style transfer in process"
python test.py --content $BASE'/data/content' \
		--style $BASE'/data/style' \
		--output_name $BASE'/output'\
		--model_state_path $BASE'/model_state.pth'\
		--n_cluster "9"\
		--gpu "-1"
echo "---------------------------------------"
echo "Done !"
