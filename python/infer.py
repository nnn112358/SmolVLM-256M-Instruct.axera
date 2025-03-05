from transformers import AutoTokenizer, AutoConfig
import numpy as np
from ml_dtypes import bfloat16
from axengine import InferenceSession
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch
from transformers.tokenization_utils_base import AddedToken
import onnxruntime as ort


def post_process(data, topk=1, topp=0.9, temperature=0.6):
    def top_p(l: np.ndarray, p: float) -> np.ndarray:
        index = np.argsort(l)
        res = l.copy()
        sum_p = 0
        for i in index[::-1]:
            if sum_p >= p:
                res[i] = 0
            sum_p += res[i]
        return res / sum_p

    def softmax(l: np.ndarray) -> np.ndarray:
        l_max = l - l.max()
        l_exp = np.exp(l_max)
        res = l_exp / np.sum(l_exp)
        return res.astype(np.float64)

    r = data.astype(np.float32)
    r = r.flatten()
    # topk
    candidate_index = np.argpartition(r, -topk)[-topk:]
    candidate_value = r[candidate_index]
    # temperature
    candidate_value /= temperature
    # softmax
    candidate_soft = softmax(candidate_value)
    # topp
    candidate_soft = top_p(candidate_soft, topp)
    candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    pos = np.random.multinomial(1, candidate_soft).argmax()
    next_token = candidate_index[pos]
    return next_token, candidate_index, candidate_soft


IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def _prompt_split_image(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_img_token,
):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}"
                + f"<row_{n_h + 1}_col_{n_w + 1}>"
                + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(
    image_seq_len, fake_token_around_image, image_token, global_img_token
):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows,
    image_cols,
    image_seq_len,
    fake_token_around_image,
    image_token,
    global_img_token,
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_img_token=global_img_token,
        )
    return _prompt_split_image(
        image_seq_len,
        image_rows,
        image_cols,
        fake_token_around_image,
        image_token,
        global_img_token,
    )


if __name__ == "__main__":
    ckpt_dir = "../SmolVLM-256M-Instruct-AX650"

    cfg = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
    # print(cfg)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)

    # load_image
    image_path = "../assets/demo.jpg"
    pixel_values = load_image(image_path, input_size=512, max_num=1)

    print("preprocess image done!")

    # extract img feature by vit
    vit_session = ort.InferenceSession(
        f"{ckpt_dir}/SmolVLM-256M-Instruct_vision.onnx",
        providers=["CPUExecutionProvider"],
    )

    vit_output = vit_session.run(
        ["last_hidden_state"], {"pixel_values": pixel_values.numpy()}
    )[0]

    print("vit feature extract done!")

    text = [
        "<|im_start|>User:<image>Can you describe this image?<end_of_utterance>\nAssistant:"
    ]
    image_rows = [[0]]
    image_cols = [[0]]
    image_seq_len = 64
    image_token = "<image>"
    fake_image_token = "<fake_token_around_image>"
    global_img_token = "<global-img>"
    prompt_strings = []
    for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
        # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
        image_prompt_strings = []
        for n_rows, n_cols in zip(sample_rows, sample_cols):
            image_prompt_string = get_image_prompt_string(
                n_rows,
                n_cols,
                image_seq_len,
                image_token=image_token,
                fake_token_around_image=fake_image_token,
                global_img_token=global_img_token,
            )
            image_prompt_strings.append(image_prompt_string)

        split_sample = sample.split(image_token)
        if len(split_sample) == 0:
            raise ValueError("The image token should be present in the text.")

        # Place in the image prompt strings where the image tokens are
        sample = split_sample[0]
        for i, image_prompt_string in enumerate(image_prompt_strings):
            sample += image_prompt_string + split_sample[i + 1]
        prompt_strings.append(sample)

    fake_image_token = AddedToken(
        fake_image_token, normalized=False, special=True
    )
    image_token = AddedToken(image_token, normalized=False, special=True)
    end_of_utterance_token = AddedToken(
        "<end_of_utterance>", normalized=False, special=True
    )
    tokens_to_add = {
        "additional_special_tokens": [
            fake_image_token,
            image_token,
            end_of_utterance_token,
        ]
    }
    tokenizer.add_special_tokens(tokens_to_add)

    token_ids = tokenizer(prompt_strings)["input_ids"][0]
    image_start_index = np.where(np.array(token_ids) == cfg.image_token_id)[0].tolist()[
        0
    ]
    image_insert_index = image_start_index + 1
    embeds = np.load(f"{ckpt_dir}/model.embed_tokens.weight.npy")
    prefill_data = np.take(embeds, token_ids, axis=0)
    prefill_data = prefill_data.astype(bfloat16)
    prefill_data[
        image_insert_index : image_insert_index + vit_output.shape[1]
    ] = vit_output[0, :, :]
    token_len = len(token_ids)

    cfg = cfg.text_config
    lastN = 1023
    kv_dim = cfg.hidden_size // cfg.num_attention_heads * cfg.num_key_value_heads
    k_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]
    v_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]

    prefill_decoder_sessins = []
    for i in range(cfg.num_hidden_layers):
        session = InferenceSession.load_from_model(
            f"{ckpt_dir}/llama_p320_l{i}_together.axmodel"
        )
        prefill_decoder_sessins.append(session)
    post_process_session = InferenceSession.load_from_model(
        f"{ckpt_dir}/llama_post.axmodel"
    )
    print("model load done!")

    """
        prefill
    """
    prefill_len = 320
    for i in range(cfg.num_hidden_layers):
        prefill_decoder_sessins[i].set_runtime_context(group_id=1)

    if prefill_len > 0:
        indices = np.array(list(range(prefill_len)), np.uint32).reshape(
            (1, prefill_len)
        )
        indices[:, token_len:] = 0
        mask = np.zeros((1, prefill_len, prefill_len)) - 65536
        data = np.zeros((1, prefill_len, cfg.hidden_size)).astype(bfloat16)
        data[:, 0:token_len] = prefill_data
        for i, t in enumerate(token_ids):
            mask[:, i, : i + 1] = 0
        mask = mask.astype(bfloat16)
        for i in range(cfg.num_hidden_layers):
            input_feed = {
                "K_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                "V_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = prefill_decoder_sessins[i].run(input_feed)
            k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
            v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
            data = outputs[2][:, :token_len, :]

    post_out = post_process_session.run({"input": data[:, token_len - 1, :]})[0]
    next_token, posssible_tokens, possible_soft = post_process(post_out, topk=1)
    posibles = [tokenizer.decode([t]) for t in posssible_tokens]
    posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
    token_ids.append(next_token)
    print("prefill done!")

    # set to decoder
    for i in range(cfg.num_hidden_layers):
        prefill_decoder_sessins[i].set_runtime_context(group_id=0)

    mask = np.zeros((1, 1, lastN + 1), dtype=np.float32).astype(bfloat16)
    mask[:, :, :lastN] -= 65536
    mask[:, :, :token_len] = 0
    for start_indice in range(lastN + 1):
        if prefill_len > 0 and start_indice < token_len:
            continue
        next_token = token_ids[start_indice]
        indices = np.array([start_indice], np.uint32).reshape((1, 1))
        data = embeds[next_token, :].reshape((1, 1, cfg.hidden_size)).astype(bfloat16)

        for i in range(cfg.num_hidden_layers):
            input_feed = {
                "K_cache": k_caches[i],
                "V_cache": v_caches[i],
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = prefill_decoder_sessins[i].run(input_feed)
            k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
            v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
            data = outputs[2]
        mask[..., start_indice] = 0
        if start_indice < token_len - 1:
            pass
        else:
            post_out = post_process_session.run({"input": data})[0]
            next_token, posssible_tokens, possible_soft = post_process(post_out)
            token_ids.append(next_token)
        if next_token == tokenizer.eos_token_id:
            # print("hit eos!")
            break
    print(tokenizer.decode(token_ids[token_len:]))
