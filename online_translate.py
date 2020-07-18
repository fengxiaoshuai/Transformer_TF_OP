#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  :  Feng Shuai

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import os
import importlib
from tensorflow.python.util import nest
from online_model import *


try:
  trt = importlib.import_module("tensorflow.contrib.tensorrt")
except:
  trt = None

from nltool.helper import pad_multi_lines

from aitranslation.backend.utils.tensorflow import *
from aitranslation.config import *
from aitranslation.backend.ctriptranslator.model_config import *
from tensor2tensorextend.utils.text_encoder import SPMTextEncoder

class Ctriptranslator(TranslatorBase):
  def __init__(self, deploy_hparam, model_config):
    super(Ctriptranslator, self).__init__(deploy_hparam, model_config)

  def init(self, deploy_hparam, model_config):
      self.is_deploy = deploy_hparam.get('is_deploy') or False
      self.is_gpu = deploy_hparam.get('is_gpu') or False
      self.model_config = model_config
      self.checkpoint_dir = os.path.expanduser(
        os.path.join(deploy_hparam.get('tpath') or "./translation_model", self.model_config.checkpoint_dir))
      self.vocab_path = os.path.join(self.checkpoint_dir, self.model_config.vocab_path)
      self.langs = self.model_config.langs
      self.encoders = SPMTextEncoder(self.vocab_path)
      logger.info("{} - Initializing...".format(self.info))
      self.warmuped = False
      logger.info("{} - Building...".format(self.info))
      #self.ckpt_path = tf.train.latest_checkpoint(self.checkpoint_dir)
      #logger.info("{} - Checkpoint:{}".format(self.info, self.ckpt_path))
      self.new_build()
      logger.info("{} - Initialization Done...".format(self.info))

  def new_build(self):
      self.out = build_model(self.checkpoint_dir)
      self.sess = tf.Session()
  def get_decode_length(self, max, scale, length):
      tem = scale*length
      if max <= tem:
        tem = max
      return tem
  def get_padding(self, inputs):
      pad = []
      for tem in inputs:
        for i in tem:
            if i == 0:
                pad.append(0)
            else:
                pad.append(1)
      return pad

  @property
  def info(self):
    return "Backend:Tensor2tensor Model:{}...".format("transformer")

  def translate(self, inputs, lang1, lang2, task_space, max_decode_length=None, decode_scale=None, beam_size=None,
                alpha=0., beta=0.):
    '''
    Translate the input sentence
    '''
    try:
      lang1 = [self.model_config.langs.index(l) for l in lang1]
      lang2 = [self.model_config.langs.index(l) for l in lang2]
      inputs_idxs = [self.encoders.encode(input_) + [self.model_config.EOS] for input_ in inputs]
      inputs_lists = [self.encoders.decode_list(encoded_input_idxs, True) for encoded_input_idxs in inputs_idxs]
      padded_inputs_idxs = pad_multi_lines(inputs_idxs, self.model_config.PAD)
      print("********************padded_inputs_idxs*************", padded_inputs_idxs)
    except:
      return [
        TranslationCandResult([], inp, None, None, None, None,
                              TranslationSetting(beam_size, alpha if beam_size else None,
                                                 beta if beam_size else None, None), None, 0.,
                              Status.report(Status.BackendError, "Translator.translate Prepare Error"))
        for inp, ts in zip(inputs, task_space)
      ]
    try:
      t0 = time.time()
      options = tf.RunOptions()
      if self.warmuped:
        options.timeout_in_ms = switch.get_back_switch("timeout", None, 15000)
      #tf_result = self.sess.run(self.translate_outputs.outputs_dict(beam_size),
      #                          feed_dict=self.translate_inputs.feed_dict(
      #                            padded_inputs_idxs,
      #                            lang1, lang2, task_space, max_decode_length,
      #                            decode_scale, beam_size, alpha, beta), options=options)
      ##############################new  code################
      pad = self.get_padding(padded_inputs_idxs)
      length = self.get_decode_length(max_decode_length, decode_scale, len(padded_inputs_idxs[0]))
      input_word = tf.get_default_graph().get_tensor_by_name("input_word:0")
      target_language = tf.get_default_graph().get_tensor_by_name("target_language:0")
      mask = tf.get_default_graph().get_tensor_by_name("mask:0")
      decode_length = tf.get_default_graph().get_tensor_by_name("decode_length:0")
      result = self.sess.run(self.out, feed_dict={input_word:padded_inputs_idxs, target_language:lang2, mask:pad, decode_length:length})
      print(result)
      ##############################new  code################
      elapsed = time.time() - t0

    except tf.errors.DeadlineExceededError:
      return [
        TranslationCandResult([], inp, None, inp_l, None, None,
                              TranslationSetting(beam_size, alpha if beam_size else None,
                                                 beta if beam_size else None, None), None, 0.,
                              Status.report(Status.TimeoutError, "Translator.translate Session Timeout"))
        for inp, inp_l, ts in zip(inputs, inputs_lists, task_space)
      ]
    except:
      return [
        TranslationCandResult([], inp, None, inp_l, None, None,
                              TranslationSetting(beam_size, alpha if beam_size else None,
                                                 beta if beam_size else None, None), None, 0.,
                              Status.report(Status.BackendError, "Translator.translate Session Run Error"))
        for inp, inp_l, ts in zip(inputs, inputs_lists, task_space)
      ]
    try:
      #max_length = tf_result["max_length"]
      targets_idxs = [result]#tf_result["targets"]
      targsts = []
      targsts_lists = []
      for targets_beam_idxs in targets_idxs:
        targsts_beam = []
        targsts_lists_beam = []
        for target_idx in targets_beam_idxs:
          target = self.encoders.decode(target_idx, True)
          target_list = self.encoders.decode_list(target_idx, True)
          if len(target_list) == 0 or target == 0: continue
          targsts_beam.append(target)
          targsts_lists_beam.append(target_list)
        if len(targsts_beam) == 0:
          targsts_beam = [""]
          targsts_lists_beam = [[]]
        targsts.append(targsts_beam)
        targsts_lists.append(targsts_lists_beam)
      #print("*****max*********", max_length)
      print("*****result*********", targets_idxs)
      return targsts
    except:
      return [
        TranslationCandResult([], inp, None, inp_l, None, None,
                              TranslationSetting(beam_size, alpha if beam_size else None,
                                                 beta if beam_size else None, None), None, 0.,
                              Status.report(Status.BackendError, "Translator.translate Result Extraction Error"))
        for inp, inp_l, ts in zip(inputs, inputs_lists, task_space)
      ]
    return [
      TranslationCandResult([], inp, trgs, inp_l, trgs_l, score,
                            TranslationSetting(beam_size, alpha if beam_size else None, beta if beam_size else None,
                                               ml), None, elapsed)
      for inp, inp_l, trgs, trgs_l, score, ml, ts in
      zip(inputs, inputs_lists, targsts, targsts_lists, scores, max_length, task_space)
    ]


if __name__ == "__main__":
  t = Ctriptranslator({'tpath': '/opt/app/translation_model', 'ldpath': '/opt/app/language_detection_model', 'is_deploy': False, 'is_gpu': False},ModelSpecEnJaKoZh())
  lang1=["zh","zh"]
  lang2=["ja","ja"]
  input= ['我','警察直升机在各个区域上方飞行了大约一个小时-没有成功。']
  result = t.translate(input, lang1, lang2, [[0]], max_decode_length=1000,
                       decode_scale=3)
  print(result)
