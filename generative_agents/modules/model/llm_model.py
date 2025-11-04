"""generative_agents.model.llm_model"""

import time
import re
import requests


class LLMModel:
    def __init__(self, config):
        self._api_key = config["api_key"]
        self._base_url = config["base_url"]
        self._model = config["model"]
        self._meta_responses = []
        self._summary = {"total": [0, 0, 0]}

        self._handle = self.setup(config)
        self._enabled = True

    def setup(self, config):
        raise NotImplementedError(
            "setup is not support for " + str(self.__class__)
        )

    def completion(
        self,
        prompt,
        retry=10,
        callback=None,
        failsafe=None,
        caller="llm_normal",
        **kwargs
    ):
        response, self._meta_responses = None, []
        self._summary.setdefault(caller, [0, 0, 0])
        for _ in range(retry):
            try:
                meta_response = self._completion(prompt, **kwargs).strip()
                self._meta_responses.append(meta_response)
                self._summary["total"][0] += 1
                self._summary[caller][0] += 1
                if callback:
                    response = callback(meta_response)
                else:
                    response = meta_response
            except Exception as e:
                print(f"LLMModel.completion() caused an error (attempt {_+1}/{retry}): {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
                response = None
                continue
            if response is not None:
                break
        pos = 2 if response is None else 1
        self._summary["total"][pos] += 1
        self._summary[caller][pos] += 1
        return response or failsafe

    def _completion(self, prompt, **kwargs):
        raise NotImplementedError(
            "_completion is not support for " + str(self.__class__)
        )

    def is_available(self):
        return self._enabled  # and self._summary["total"][2] <= 10

    def get_summary(self):
        des = {}
        for k, v in self._summary.items():
            des[k] = "S:{},F:{}/R:{}".format(v[1], v[2], v[0])
        return {"model": self._model, "summary": des}

    def disable(self):
        self._enabled = False

    @property
    def meta_responses(self):
        return self._meta_responses


class OpenAILLMModel(LLMModel):
    def setup(self, config):
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _completion(self, prompt, temperature=0.5):
        messages = [{"role": "user", "content": prompt}]
        response = self._handle.chat.completions.create(
            model=self._model, messages=messages, temperature=temperature
        )
        if len(response.choices) > 0:
            return response.choices[0].message.content
        return ""


class OllamaLLMModel(LLMModel):
    def setup(self, config):
        return None

    def ollama_chat(self, messages, temperature):
        headers = {
            "Content-Type": "application/json"
        }
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        # Set timeout to prevent hanging (30 seconds for connect, 300 seconds for read)
        timeout = (30, 300)
        try:
            response = requests.post(
                url=f"{self._base_url}/chat/completions",
                headers=headers,
                json=params,
                stream=False,
                timeout=timeout
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.Timeout as e:
            print(f"Request to Ollama timed out: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request to Ollama failed: {e}")
            raise
        except ValueError as e:
            print(f"Failed to parse JSON response from Ollama: {e}")
            raise

    def _completion(self, prompt, temperature=0.5):
        if "qwen3" in self._model and "\n/nothink" not in prompt:
            # Disable think for Qwen3 model to improve inference speed
            prompt += "\n/nothink"
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.ollama_chat(messages=messages, temperature=temperature)
            if response and "choices" in response and len(response["choices"]) > 0:
                ret = response["choices"][0]["message"]["content"]
                # Filter out <think> tags to avoid affecting subsequent logic
                return re.sub(r"<think>.*</think>", "", ret, flags=re.DOTALL)
            else:
                print(f"Warning: Invalid response format from Ollama: {response}")
                return ""
        except Exception as e:
            print(f"Error in Ollama _completion: {e}")
            raise


def create_llm_model(llm_config):
    """Create llm model"""

    if llm_config["provider"] == "ollama":
        return OllamaLLMModel(llm_config)

    elif llm_config["provider"] == "openai":
        return OpenAILLMModel(llm_config)
    else:
        raise NotImplementedError(
            "llm provider {} is not supported".format(llm_config["provider"])
        )
    return None


def parse_llm_output(response, patterns, mode="match_last", ignore_empty=False):
    if isinstance(patterns, str):
        patterns = [patterns]
    rets = []
    for line in response.split("\n"):
        line = line.replace("**", "").strip()
        for pattern in patterns:
            if pattern:
                matchs = re.findall(pattern, line)
            else:
                matchs = [line]
            if len(matchs) >= 1:
                rets.append(matchs[0])
                break
    if not ignore_empty:
        assert rets, "Failed to match llm output"
    if mode == "match_first":
        return rets[0]
    if mode == "match_last":
        return rets[-1]
    if mode == "match_all":
        return rets
    return None
