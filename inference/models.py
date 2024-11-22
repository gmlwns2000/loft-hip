# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Models used for inference."""

import abc
import enum
from typing import Any, List

from absl import logging
from inference import utils
import vertexai
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmCategory
from vertexai.generative_models import Part
from vertexai.generative_models import SafetySetting


ContentChunk = utils.ContentChunk
MimeType = utils.MimeType
LOCATION = 'us-central1'
TEMPERATURE = 0.0


class GeminiModel(enum.Enum):
  GEMINI_1_5_FLASH_002 = 'gemini-1.5-flash-002'  # Max input tokens: 1,048,576
  GEMINI_1_5_PRO_002 = 'gemini-1.5-pro-002'  # Max input tokens: 2,097,152
  OPENAI = 'openai'


class Model(metaclass=abc.ABCMeta):
  """Base class for models."""

  def index(
      self,
      content_chunks: List[ContentChunk],
      document_indices: List[tuple[int, int]],
      **kwargs: Any,
  ) -> str:
    """Indexes the example containing the corpus.

    Arguments:
      content_chunks: list of content chunks to send to the model.
      document_indices: list of (start, end) indices marking the documents
        boundaries within content_chunks.
      **kwargs: additional arguments to pass.

    Returns:
      Indexing result.
    """
    del content_chunks, document_indices, kwargs  # Unused.
    return 'Indexing skipped since not supported by model.'

  @abc.abstractmethod
  def infer(
      self,
      content_chunks: List[ContentChunk],
      document_indices: List[tuple[int, int]],
      **kwargs: Any,
  ) -> str:
    """Runs inference on model and returns text response.

    Arguments:
      content_chunks: list of content chunks to send to the model.
      document_indices: list of (start, end) indices marking the documents
        boundaries within content_chunks.
      **kwargs: additional arguments to pass to the model.

    Returns:
      Inference result.
    """
    raise NotImplementedError


class VertexAIModel(Model):
  """GCP VertexAI wrapper for general Gemini models."""

  def __init__(
      self,
      project_id: str,
      model_name: str,
      pid_mapper: dict[str, str],
  ):
    self.project_id = project_id
    self.model_name = model_name
    self.pid_mapper = pid_mapper
    vertexai.init(project=project_id, location=LOCATION)
    self.model = GenerativeModel(self.model_name)

  def _process_content_chunk(self, content_chunk: ContentChunk) -> Part:
    if content_chunk.mime_type in [
        MimeType.TEXT,
        MimeType.IMAGE_JPEG,
        MimeType.AUDIO_WAV,
    ]:
      return Part.from_data(
          content_chunk.data, mime_type=content_chunk.mime_type
      )
    else:
      raise ValueError(f'Unsupported MimeType: {content_chunk.mime_type}')

  def _get_safety_settings(
      self, content_chunks: List[ContentChunk]
  ) -> List[SafetySetting]:
    """Returns safety settings for the given content chunks."""
    # Audio prompts cannot use BLOCK_NONE.
    if any(
        content_chunk.mime_type == MimeType.AUDIO_WAV
        for content_chunk in content_chunks
    ):
      threshold = SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
    else:
      threshold = SafetySetting.HarmBlockThreshold.BLOCK_NONE
    return [
        SafetySetting(
            category=category,
            threshold=threshold,
        )
        for category in HarmCategory
    ]

  def _postprocess_response(self, response: Any) -> List[str]:
    """Postprocesses the response from the model."""
    try:
      output_text = getattr(response, 'candidates')[0].content.parts[0].text
      final_answers = utils.extract_prediction(output_text)
      final_answers = [
          self.pid_mapper[str(answer)] for answer in final_answers
      ]
    except Exception as e:  # pylint:disable=broad-exception-caught
      logging.error('Bad response %s with error: %s', response, str(e))
      raise ValueError(f'Unexpected response: {response}') from e

    return final_answers

  def infer(
      self,
      content_chunks: List[ContentChunk],
      **kwargs: Any,
  ) -> List[str]:
    response = self.model.generate_content(
        [
            self._process_content_chunk(content_chunk)
            for content_chunk in content_chunks
        ],
        generation_config=GenerationConfig(temperature=TEMPERATURE, top_p=1.0),
        safety_settings=self._get_safety_settings(content_chunks),
    )

    return self._postprocess_response(response)

class OpenAiModel(Model):
  def __init__(self, pid_mapper):
    self.pid_mapper = pid_mapper
  
  def infer(
    self, 
    content_chunks: List[ContentChunk], 
    **kwargs
  ):
    import requests, os
    
    lines = list(map(lambda x: x._text, content_chunks))
    # for i in range(len(lines)):
    #   if lines[i].startswith('ID:'):
    #     spl = lines[i].split('|')
    #     header = spl[0].strip()
    #     header = f'# Start of ID: \'{header.strip("ID: ")}\''
    #     title = spl[1].strip()
    #     content = '|'.join(spl[2:-1]).strip()
    #     footer = spl[-1].strip()
    #     footer = f'# The end of ID: \'{footer.strip("END ID: ")}\''
    #     lines[i] = f'-----\n\n{header}\n\n**{title}**\n\n> {content}\n\n{footer}\n\n-----'
    # prompt = "\n\n".join(lines)
    prompt = "\n".join(lines)
    # prompt = map(lambda line: {'role':'user', 'content':line}, lines)
    
    # print(prompt, end='\n\n')
    print('>>>', prompt[:50].replace('\n','\\n'), '...', prompt[-50:].replace('\n','\\n'), '<<<')
    print('Now, wait for response...', flush=True)
    
#     IS_GEMMA = os.getenv('IS_GEMMA', '0') == '1'
#     if IS_GEMMA:
#       pass
#     else:
#       prompt = f"""<|start_header_id|>system<|end_header_id|>

# Cutting Knowledge Date: December 2023
# Today Date: 26 Jul 2024

# <|eot_id|><|start_header_id|>user<|end_header_id|>

# {prompt}

# <|eot_id|><|start_header_id|>assistant<|end_header_id|>

# """    
    endpoint = os.getenv('ENDPOINT', 'http://localhost:30000/v1')
    api_key = os.getenv('API_KEY', 'sk-dummy')
    response = requests.post(
      f"{endpoint}/chat/completions",
      headers={"Authorization": f"Bearer {api_key}"},
      json={
        "model": "any", 
        "max_tokens": 256,
        # "min_tokens": 16, # THIS HURT PERFORMANCE ALOT.
        # "top_k": 1,
        "temperature": 0,
        "messages": [
          {
          'role': 'user',
          'content': prompt,
          }
        ]
      },
    )
    
    # print(response.json())
    
    assert response.status_code == 200
    try:
      text = response.json()['choices'][0]['message']['content']
    except:
      text = response.json()['choices'][0]['text']
    text = text.replace('[|endofturn|]', '')
    text = text.replace('<|eot_id|>', '')
    text = text.replace('<end_of_turn>', '')
    print('Generated:', text.replace('\n','\\n'))
    
    final_answers = utils.extract_prediction(text)
    if os.getenv('IGNORE_PID_MAPPER', '0') == '0':
      final_answers = [
        self.pid_mapper[str(answer)] if str(answer) in self.pid_mapper else str(answer) for answer in final_answers
      ]
    else:
      final_answers = [text]
    
    print('Final Answer:', final_answers, flush=True)
    
    return final_answers

def get_model(
    model_url_or_name: str,
    project_id: str | None,
    pid_mapper: dict[str, str],
) -> Model:
  """Returns the model to use."""

  if model_url_or_name in map(lambda x: x.value, GeminiModel.__members__.values()):
    if model_url_or_name == GeminiModel.OPENAI.value:
      model = OpenAiModel(
        pid_mapper=pid_mapper,
      )
    else:
      if project_id is None:
        raise ValueError(
            'Project ID and service account are required for VertexAIModel.'
        )
      model = VertexAIModel(
          project_id=project_id,
          model_name=model_url_or_name,
          pid_mapper=pid_mapper,
      )
  else:
    raise ValueError(f'Unsupported model: {model_url_or_name}')
  return model
