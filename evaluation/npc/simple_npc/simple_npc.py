import base64
import random
from .prompt import TEMPLATE, DISAMBIGUATION_PROMPT
class SimpleNPC:
    def __init__(
        self,
        max_interaction_turn: int,
        name: str = 'openai',
        model_name: str = None,
        openai_api_key: str = None,
        base_url: str = None,
    ) -> None:
        self.name = name
        if self.name in ['openai', 'deepseek']:
            try:
                from openai import OpenAI
            except ModuleNotFoundError:
                print('ModuleNotFoundError: No module named \'openai\'. Please install it first.')
                return
            self.model_name = model_name
            self.history_messages = []
            with open(openai_api_key, 'r', encoding='utf-8') as file:
                openai_api_key = file.read().strip()
            try:
                self.llm = OpenAI(api_key=openai_api_key, base_url=base_url)
            except Exception as e:
                print(f'Failed to initialize OpenAI: {e}')
        elif self.name == 'qwen':
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(
                '/mnt/inspurfs/efm_t/zhushaohao/init_checkpoint/Qwen/Qwen3-VL-4B-Instruct',
                trust_remote_code=True,
            )

            self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
                '/mnt/inspurfs/efm_t/zhushaohao/init_checkpoint/Qwen/Qwen3-VL-4B-Instruct',
                dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

            self.llm.eval()
        else:
            raise ValueError(f"Invalid model name: {self.name}")
        self.max_turn = max_interaction_turn


    def get_room_name(self, room):
        room_name_dict = {
            "living region": "living room",
            "stair region": "stairs",
            "bathing region": "bathroom",
            "storage region": "storage room",
            "study region": "study room",
            "cooking region": "kitchen",
            "sports region": "sports room",
            "corridor region": "corridor",
            "toliet region": "toilet",
            "dinning region": "dining room",
            "resting region": "resting room",
            "open area region": "open area",
            "other region": "area",
        }
        return room_name_dict[room]

    def answer_question(self, question: str, instance_id: str,  object_dict: dict, task_done: bool, path_description: str, mode: str, forbidden: list = []):
        if mode == 'one_turn':
            goal_information = ''
            goal_information += 'room: ' + self.get_room_name(object_dict[instance_id]['room']) + '\n'
            goal_information += '\n'.join([f'{a.lower()}: {i.lower()}' for a,i in object_dict[instance_id]['unique_description'].items() if a in ['color','texture','material','shape','placement'] and len(i)>0])
            nearby_objects = [object_dict[obj]['unique_description']['fine grained category'].lower() for obj, _ in object_dict[instance_id]['nearby_objects'].items() if obj in object_dict and isinstance(object_dict[obj]['unique_description'], dict)]
            if len(nearby_objects) > 0:
                goal_information += '\nnearby objects: ' + ','.join(nearby_objects)
            goal_information += 'whole description: ' + object_dict[instance_id]['caption']
            answer = self.ask_directly(
                template_type="one_turn_prompt",
                question=question,
                goal_information = goal_information,
                path_description = path_description,
                task_done = task_done,
            )
            return answer
        elif mode == 'two_turn':
            answer = self.ask_directly(
                template_type="two_turn_prompt_0",
                question=question,
            )
            if 'path' in answer.lower():
                if 'path' in forbidden:
                    return "I don't know."
                return path_description
            elif 'disambiguation' in answer.lower():
                if 'disambiguation' in forbidden:
                    return "I don't know."
                if task_done:
                    return random.choice(DISAMBIGUATION_PROMPT['yes'])
                else:
                    return random.choice(DISAMBIGUATION_PROMPT['no'])
            elif 'information' in answer.lower():
                if 'information' in forbidden:
                    return "I don't know."
                goal_information = ''
                goal_information += 'room: ' + self.get_room_name(object_dict[instance_id]['room']) + '\n'
                goal_information += '\n'.join([f'{a.lower()}: {i.lower()}' for a,i in object_dict[instance_id]['unique_description'].items() if a in ['color','texture','material','shape','placement'] and len(i)>0])
                nearby_objects = [object_dict[obj]['unique_description']['fine grained category'].lower() for obj, _ in object_dict[instance_id]['nearby_objects'].items() if obj in object_dict and isinstance(object_dict[obj]['unique_description'], dict)]
                if len(nearby_objects) > 0:
                    goal_information += '\nnearby objects: ' + ','.join(nearby_objects)
                goal_information += 'whole description: ' + object_dict[instance_id]['caption']
                answer = self.ask_directly(
                    template_type="one_turn_prompt",
                    question=question,
                    goal_information = goal_information,
                    path_description = path_description,
                    task_done = task_done,
                )
                answer = self.answer_question(question, instance_id, object_dict, task_done, answer, 'one_turn')
            return answer
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def ask_directly(self, template_type, **kwargs):
        def generate_prompt(template_type, **kwargs):
            """
            Generate a complete prompt based on the template type and provided content.

            Parameters:
                template_type (str): The type of template to use.
                **kwargs: The content to fill in the template.

            Returns:
                str: The complete prompt.
            """

            prompt = TEMPLATE.get(template_type, None)
            if prompt is None:
                raise ValueError(f"Template type '{template_type}' not found.")
            prompt = prompt.format(**kwargs)
            return prompt
        messages = []
        image_bufs = kwargs.get('images', None)
        cnt = 0
        prompt = generate_prompt(template_type, **kwargs)
        content = [{'type': 'text', 'text': prompt}]
        if image_bufs is not None:
            for im_id, image_buf in enumerate(image_bufs):
                img_encoded = base64.b64encode(image_buf.getvalue()).decode('utf-8')
                image_buf.close()
                item = {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{img_encoded}',
                        'detail': 'high',
                    },
                    'index': im_id,
                }
                content.append(item)
        messages.append({'role': 'user', 'content': content})

        while cnt < self.max_turn:
            try:
                if self.name in ['openai', 'deepseek']:
                    response = self.llm.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=2048,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    result = response.choices[0].message.content
                elif self.name == 'qwen':
                    inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )

                    inputs = inputs.to(self.llm.device)

                    # 有些 transformers / processor 版本会生成 token_type_ids，
                    # Qwen3-VL generate 时可能不需要它
                    inputs.pop("token_type_ids", None)

                    generated_ids = self.llm.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                    )

                    # 去掉 prompt，只保留模型新生成的内容
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]

                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    result = output_text[0]
                else:
                    raise ValueError(f"Invalid model name: {self.name}")
                break
            except Exception as e:
                print(e)
                cnt += 1
                result = None
        return result
