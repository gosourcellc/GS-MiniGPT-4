from PIL import Image
from fast_api.exceptions import MiniGPTException
from tempfile import NamedTemporaryFile
from minigpt4 import MiniGPT4
from minigpt4.common.registry import registry
from minigpt4.common.config import Config
from minigpt4.conversation.conversation import (
    Chat,
    CONV_VISION_Vicuna0,
    CONV_VISION_LLama2,
)
from argparse import Namespace
from typing import Optional


DEFAULT_CONFIG_PATH = "eval_configs/minigpt4_eval.yaml"
DEFAULT_GPU_ID = 0


class MiniGPT:
    is_initialized: bool = False
    chat: Chat
    vis_processor: Chat
    model: MiniGPT4
    CONV_VISION: dict
    conv_dict = {
        "pretrain_vicuna0": CONV_VISION_Vicuna0,
        "pretrain_llama2": CONV_VISION_LLama2,
    }

    @staticmethod
    def get_chat_arguments(
        cfg_path: Optional[str] = DEFAULT_CONFIG_PATH,
        gpu_id: Optional[int] = DEFAULT_GPU_ID,
        options: Optional[list] = None,
        **kwargs,
    ) -> Namespace:
        return Namespace(cfg_path=cfg_path, gpu_id=gpu_id, options=options, **kwargs)

    def setup(self, args: Namespace):
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(
            "cuda:{}".format(args.gpu_id)
        )
        self.CONV_VISION = self.conv_dict[model_config.model_type]
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)
        self.chat = Chat(
            self.model, self.vis_processor, device="cuda:{}".format(args.gpu_id)
        )
        self.is_initialized = True

    def prompt_image(
        self,
        prompt: str,
        image_file: NamedTemporaryFile,
        temperature: float,
        num_beams: int,
    ):
        if not self.is_initialized:
            raise MiniGPTException("Chat is not initialized")
        if not image_file:
            raise MiniGPTException("Image file is required")
        if not prompt:
            raise MiniGPTException("Prompt is required")

        chat_state = self.CONV_VISION.copy()
        img_list = []
        # with Image.open(image_file) as im:
        raw_image = Image.open(image_file.name).convert("RGB")
        _llm_message = self.chat.upload_img(raw_image, chat_state, img_list)

        self.chat.ask(prompt, chat_state)

        return self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            max_new_tokens=3000,
            num_beams=num_beams,
            temperature=temperature,
        )[0]
