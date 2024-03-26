## script to lunch the experimentations :
BASE=$(pwd)
echo "---------------------------------------"
echo "Process the data"
python3 dataset.py 	--test_content_dir $BASE'/data/content'\
			--test_style_dir $BASE'/data/style'
echo "---------------------------------------"
echo "Style transfer in process"
for i in 1 2 3 4 5 6 7 8 9 10;do
	python test.py --content $BASE'/data/content_resized/cornell.jpg' \
		--style $BASE'/data/style_resized/dots_star.jpg' \
		--output_name $BASE'/output'\
		--model_state_path $BASE'/model_state.pth'\
		--n_cluster $i\
		--gpu "-1"\
		--print_tsne "True" \
		--print_cluster_criterium "True"
done
echo "---------------------------------------"
echo "Done !"
