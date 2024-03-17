## script to lunch the experimentations :

echo "Process the data"
!python3 dataset.py 	--train_content_dir\
			--train_style_dir\
			--test_content_dir '/content/data/content/test2017'\
			--test_style_dir '/content/data/style/test'
echo "Style transfer in process"
!python test.py --content 'data/content' \
		--style 'data/style' \
		--output_name 'output'	\
		--model_state_path 'model_state.pth' \
		--n_cluster "9"\
		--gpu "-1" 

echo "Done !"
