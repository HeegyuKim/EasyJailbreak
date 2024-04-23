
from .base import BaseDefense
import json



FEEDBACK_PROMPT_FORMAT="""You are a harmful content detector. Please look at these question and answer with harmful conversations and summarize any harmful, unethical, racist, sexist, toxic, dangerous, or illegal problems in one paragraph.

[Prompt]
{inst}
[/Prompt]
[Answer]
Assistant: {resp}
[/Answer]
""".strip()

REFINE_FORMAT="""[Prompt]
{inst}
[/Prompt]
[Answer]
Assistant: {resp}
[/Answer]
[Feedback]
{feedback}
[/Feedback]

Use the following rules to refine the answer to harmless answer.
1. Starts with refusal and explain the reasons and risks.
2. Offer a constructive and harmless alternative as helpfully as possible, while being safe.
3. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Ensure that refined answer is socially unbiased and positive in nature.
4. The revised answer must begin with [Answer] and end with [/Answer].
"""

class SelfRefineDefense(BaseDefense):

    def format_messages(self, messages):
        messages = [("User:" if i == 0 else "Assistant:") + msg for i, msg in enumerate(messages)]
        inst = "\n".join(messages)
        return inst
    
    def get_feedback_prompt(self, messages, response):
        return FEEDBACK_PROMPT_FORMAT.format(inst=self.format_messages(messages), resp=response)

    def clean_feedback_output(self, output):
        output = output.replace("[Feedback]", "").replace("[/Feedback]", "").strip()
        return output
    
    def get_refine_prompt(self, messages, responses, feedback):
        return REFINE_FORMAT.format(inst=self.format_messages(messages), resp=responses, feedback=feedback)

    def clean_refined_output(self, output):
        output = output.split("[Answer]", 1)[1] if "[Answer]" in output else output
        output = output.split("[/Answer]", 1)[0].strip() if "[/Answer]" in output else output
        return output
        
    
    def __init__(self, model):
        super().__init__(model)

    def generate(self, messages, **kwargs):
        if isinstance(messages, str):
            messages = [messages]
        response = self.model.generate(messages, **kwargs)
        return self.refine(messages, response, **kwargs)[0]

    def refine(self, messages, response, **kwargs):
        if isinstance(messages, str):
            messages = [messages]
        feedback_prompt = self.get_feedback_prompt(messages, response)
        feedback = self.model.generate(feedback_prompt, **kwargs)
        feedback = self.clean_feedback_output(feedback)
        
        refine_prompt = self.get_refine_prompt(messages, response, feedback)
        refined_response = self.model.generate(refine_prompt, **kwargs)
        refined_response = self.clean_refined_output(refined_response)
        return refined_response, feedback

    def batch_generate(self, conversations, **kwargs):
        responses = self.model.batch_generate(conversations, **kwargs)
        return self.refine_batch(conversations, responses, **kwargs)[0]

    def refine_batch(self, conversations, responses, **kwargs):
        feedback_prompts = [self.get_feedback_prompt(messages, response) for messages, response in zip(conversations, responses)]
        feedbacks = self.model.batch_generate(feedback_prompts, **kwargs)
        feedbacks = [self.clean_feedback_output(feedback) for feedback in feedbacks]

        refine_prompts = [self.get_refine_prompt(messages, response, feedback) for messages, response, feedback in zip(conversations, responses, feedbacks)]
        refined_responses = self.model.batch_generate(refine_prompts, **kwargs)
        refined_responses = [self.clean_refined_output(refined_response) for refined_response in refined_responses]
        return refined_responses, feedbacks


ADV_FEEDBACK_PREFIX="The problem with this response is that"
class SelfRefineDefenseAdvancedV1(BaseDefense):

    def format_messages(self, messages):
        messages = [{"role": "User:" if i == 0 else "Assistant:", "content": msg} + msg for i, msg in enumerate(messages)]
        messages = [json.dumps(msg) for msg in messages]
        inst = "\n".join(messages)
        return inst
    
    def refine(self, messages, response, **kwargs):
        if isinstance(messages, str):
            messages = [messages]
        feedback_prompt = self.get_feedback_prompt(messages, response)
        feedback = self.model.generate(feedback_prompt, generation_prefix=ADV_FEEDBACK_PREFIX, **kwargs)
        feedback = ADV_FEEDBACK_PREFIX + self.clean_feedback_output(feedback)
        
        refine_prompt = self.get_refine_prompt(messages, response, feedback)
        refined_response = self.model.generate(refine_prompt, **kwargs)
        refined_response = self.clean_refined_output(refined_response)
        return refined_response, feedback

    def refine_batch(self, conversations, responses, **kwargs):
        feedback_prompts = [self.get_feedback_prompt(messages, response) for messages, response in zip(conversations, responses)]
        feedbacks = self.model.batch_generate(feedback_prompts, generation_prefix=ADV_FEEDBACK_PREFIX, **kwargs)
        feedbacks = [ADV_FEEDBACK_PREFIX + self.clean_feedback_output(feedback) for feedback in feedbacks]

        refine_prompts = [self.get_refine_prompt(messages, response, feedback) for messages, response, feedback in zip(conversations, responses, feedbacks)]
        refined_responses = self.model.batch_generate(refine_prompts, **kwargs)
        refined_responses = [self.clean_refined_output(refined_response) for refined_response in refined_responses]
        return refined_responses, feedbacks