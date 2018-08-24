import torchvision
import torch

import utils
import argparse

from torchvision import transforms
from vgg import Vgg19


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    style_transform = transforms.Compose(
        [transforms.ToTensor(),
         ]
    )
    content_transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    content_image = utils.load_image(args.content_image, size=args.image_size)
    style_image = utils.load_image(args.style_image, size=args.image_size)

    content = content_transform(content_image).unsqueeze(0).to(device)
    style = style_transform(style_image).unsqueeze(0).to(device)

    target = content.clone().to(device)
    target.requires_grad = True
    optimizer = torch.optim.Adam([target], lr=args.lr)

    vgg = Vgg19().to(device)

    content_features = vgg(content)
    style_features = vgg(style)

    for step in range(args.total_step):
        target_features = vgg(target)

        content_loss = 0.0
        style_loss = 0.0

        for t_f, c_f, s_f in zip(target_features, content_features, style_features):
            content_loss += args.content_weight*((t_f-c_f)**2).mean()

            t_gram = utils.gram_matrix(t_f)
            s_gram = utils.gram_matrix(s_f)

            style_loss += args.style_weight *((t_gram - s_gram)**2).mean()

        loss = content_loss + style_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % args.log_step == 0:
            print('Step [%d/%d], Content Loss: %.4f, Style Loss: %.4f, Total Loss: %.4f'
                  % (step + 1, args.total_step, content_loss.data[0], style_loss.data[0], loss.data[0]))

        if (step + 1) % args.sample_step == 0:
            img = target.clone().cpu().squeeze()
            img = img.data.clamp_(0, 1)
            torchvision.utils.save_image(img, r'.\images\out\output-%d.png' % (step + 1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_image', type=str, default=r'./images/content-images/Leonardo.jpg')
    parser.add_argument('--style_image', type=str, default=r'./images/style-images/picasso_selfport1907.jpg')
    parser.add_argument('--image_size', type=int, default=400)
    parser.add_argument('--image_scale', type = float, default=None)
    parser.add_argument('--total_step', type=int, default=2000)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--content_weight', type=float, default=1e2)
    parser.add_argument('--style_weight', type=float, default=1e10)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    print(args)
    main(args)