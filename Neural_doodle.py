import torchvision
import torch

import utils
import argparse
import numpy as np

from torchvision import transforms

from vgg import Vgg19, Vgg19_mask



def kmeans(xs, k):
    assert xs.ndim == 2
    try:
        from sklearn.cluster import k_means
        _, labels, _ = k_means(xs.astype('float64'), k)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(xs, k, missing='raise')
    return labels


def load_mask_labels(args):
    '''Load both target and style masks.
    A mask image (nr x nc) with m labels/colors will be loaded
    as a 4D boolean tensor: (1, m, nr, nc)
    '''
    target_mask_img = utils.load_image(args.target_mask_image, size=args.image_size)
    style_mask_img = utils.load_image(args.style_mask_image, size=args.image_size)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         ]
    )

    mask_vecs = np.vstack([np.array(style_mask_img).reshape((3, -1)).T,
                               np.array(target_mask_img).reshape((3, -1)).T])

    num_labels = args.num_labels
    img_nrows = args.image_size
    img_ncols = args.image_size

    labels = kmeans(mask_vecs, num_labels)
    style_mask_label = labels[:img_nrows *
                              img_ncols].reshape((img_nrows, img_ncols))
    target_mask_label = labels[img_nrows *
                               img_ncols:].reshape((img_nrows, img_ncols))

    stack_axis = 0
    style_mask = np.stack([style_mask_label == r for r in range(num_labels)],
                          axis=stack_axis)
    target_mask = np.stack([target_mask_label == r for r in range(num_labels)],
                           axis=stack_axis)

    return (np.expand_dims(style_mask, axis=0),
            np.expand_dims(target_mask, axis=0))



def region_style_loss(style_image, target_image, style_mask, target_mask):
    '''Calculate style loss between style_image and target_image,
    for one common region specified by their (boolean) masks
    '''
    masked_style = style_image * style_mask
    masked_target = target_image * target_mask

    num_channels = (style_image.shape)[0]

    s = utils.gram_matrix(masked_style) / style_mask.mean() / num_channels
    c = utils.gram_matrix(masked_target) / target_mask.mean() / num_channels

    return ((s - c)**2).mean()


def style_loss(style_image, target_image, style_masks, target_masks, args):
    '''Calculate style loss between style_image and target_image,
    in all regions.
    '''
    loss = 0.0
    for i in range(args.num_labels):

        style_mask = style_masks[i, :, :]
        target_mask = target_masks[i, :, :]

        loss += region_style_loss(style_image,
                                  target_image, style_mask, target_mask)
    return loss


def content_loss(content_image, target_image):
    return ((target_image - content_image)**2).mean()



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
    vgg_mask = Vgg19_mask().to(device)

    raw_style_mask, raw_target_mask = load_mask_labels(args)
    raw_style_mask = torch.Tensor(1.0*raw_style_mask).to(device)
    raw_target_mask = torch.Tensor(1.0*raw_target_mask).to(device)


    content_features = vgg(content)
    style_features = vgg(style)
    mask_style_features = vgg_mask(raw_style_mask)
    mask_target_features = vgg_mask(raw_target_mask)


    for step in range(args.total_step):
        target_features = vgg(target)

        con_loss = 0.0
        s_loss = 0.0

        for t_f, c_f, s_f, m_s_f, m_t_f in zip(target_features, content_features, style_features, mask_style_features, mask_target_features):

            con_loss += content_loss(t_f, c_f)
            s_loss += style_loss(s_f, t_f, m_s_f.squeeze(0), m_t_f.squeeze(0), args)

        loss = args.content_weight * con_loss + args.style_weight * s_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
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
    parser.add_argument('--style_image', type=str, default=r'./data/Monet/style.png')
    parser.add_argument('--image_size', type=int, default=100)
    parser.add_argument('--image_scale', type = float, default=None)
    parser.add_argument('--total_step', type=int, default=2000)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=20)
    parser.add_argument('--content_weight', type=float, default=1e2)
    parser.add_argument('--style_weight', type=float, default=1e10)
    parser.add_argument('--lr', type=float, default=0.01)


    parser.add_argument('--target_mask_image', type=str, default = r'./data/Monet/target_mask.png')
    parser.add_argument('--style_mask_image', type=str, default = r'./data/Monet/style_mask.png')
    parser.add_argument('--num_labels', type=int, default = 5)

    args = parser.parse_args()

    main(args)
