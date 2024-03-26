import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model
import numpy as np 

trans = transforms.Compose([transforms.ToTensor()])


def main():
    parser = argparse.ArgumentParser(description='Multimodal Style Transfer by Pytorch')
    parser.add_argument('--content', '-c', type=str, default=None,
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default=None,
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--n_cluster', type=int, default=3,
                        help='number of clusters of k-means ')
    parser.add_argument('--alpha', default=1,
                        help='fusion degree, should be a float or a list which length is n_cluster')
    parser.add_argument('--lam', type=float, default=0.1,
                        help='weight of pairwise term in alpha-expansion')
    parser.add_argument('--max_cycles', default=None,
                        help='max_cycles of alpha-expansion')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(negative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help='pretrained model state')
    parser.add_argument('--print_tsne', type=str, default='kmeans',
                        help='If t-sne should be displayed')
    parser.add_argument('--print_cluster_criterium', type=bool, default='False',
                        help='If criterium should be displayed and computed ')
    args = parser.parse_args()
    #########
    if "." not in  args.content:
        print('|_ Image not specified - random pick')
        if '_resized' not in args.content:
            content_dir_resized = args.content + '_resized'
        else : 
            content_dir_resized = args.content
        assert (os.path.exists(content_dir_resized)) , 'The content must be resized before to avoid issues'
        random_pick =np.random.choice(os.listdir(content_dir_resized), 1)[0]
        content_dir_resized = content_dir_resized + '/'+ random_pick
        print(f'|__ Image picked - {random_pick}')
    else :
        assert '_resized' in  args.content, "Image must be resized (or in a resized directory)"
        print('|_ Image specified and resized') 
        content_dir_resized =args.content

    if "." not in args.style:
        print('|_ Image not specified - random pick')
        if '_resized' not in args.style:
            style_dir_resized = args.style + '_resized'
        else : 
            style_dir_resized = args.style    
        assert (os.path.exists(style_dir_resized)), 'The style must be resized before to avoid issues'
        random_pick = np.random.choice(os.listdir(style_dir_resized), 1)[0]
        style_dir_resized = style_dir_resized +'/'+ random_pick 
        print(f'|__ Image picked - {random_pick}')

    else : 
        assert '_resized' in  args.style, "Image must be resized (or in a resized directory)"
        style_dir_resized = args.style 
        print('|_ Image specified and resized') 

    #########
    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model(n_cluster=args.n_cluster,
                  alpha=args.alpha,
                  device=device,
                  lam=args.lam,
                  max_cycles=args.max_cycles, 
                  print_tsne=args.print_tsne,
                  print_cluster_criterium=args.print_cluster_criterium)
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
        #print(f'{args.model_state_path} loaded')
        print('|_ model loaded')
    model = model.to(device)

    c = Image.open(content_dir_resized)
    s = Image.open(style_dir_resized)
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor, print_tsne=args.print_tsne, print_cluster_criterium= args.print_cluster_criterium).to('cpu')

    c_name = os.path.splitext(os.path.basename(content_dir_resized))[0]
    s_name = os.path.splitext(os.path.basename(style_dir_resized))[0]

    if args.output_name is None:
        output_name = f'/{c_name}_{s_name}_cluster_{args.n_cluster}'
    else:
        output_name = f'{args.output_name}/{c_name}_{s_name}_cluster_{args.n_cluster}'

    save_image(out, f'{output_name}.jpg', nrow=1)
    
    o = Image.open(f'{output_name}.jpg')

    demo = Image.new('RGB', (c.width * 2, c.height))
    o = o.resize(c.size)
    s = s.resize((i // 4 for i in c.size))

    demo.paste(c, (0, 0))
    demo.paste(o, (c.width, 0))
    demo.paste(s, (c.width, c.height - s.height))
    demo.save(f'{output_name}_style_transfer_demo.jpg', quality=95)

    o.paste(s,  (0, o.height - s.height))
    o.save(f'{output_name}_with_style_image.jpg', quality=95)
    
    print(f'|_ result saved {output_name}')


if __name__ == '__main__':
    main()