import copy

import data
from data.data_preprocessor import preprocess_batch
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
from test_fid import test_fid_sever  # import the fid calculator module.
from util.image import compose_retrieval_and_modified_images


def fn(data_i):
    d = preprocess_batch(
        data_i,
        False,
        src_dir=None,
        dest_dir=None
    )
    return d


if __name__ == '__main__':
    opt = TestOptions(real_image_path="").parse()
    # Initialize the server so that it can run in an subprocess with `os.popen`
    fid_calculator_sever = test_fid_sever.FidCalculatorSever(
        real_img_dir=opt.real_image_path,
        gpu_id=1,
        redirect_file=f"{opt.name}_fid_record.txt",
        best_info_path=f"{opt.name}_best_model.pkl",
        port=2345
        )

    dataloader, length = data.create_dataloader(opt, data_length=True)

    # Run the server
    fid_calculator_sever.run(
        Pix2PixModel,
        opt,
        dataloader,
        length,
        model_path=f'./checkpoints/{opt.name}/latest_net_G.pth',
        fn=fn
    )


