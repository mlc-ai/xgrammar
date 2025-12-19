Structural Tag
==========================

.. currentmodule:: xgrammar.structural_tag

This page contains the API reference for the structural tag. For its usage, see
:doc:`Structural Tag Usage <../../tutorials/structural_tag>`.


Top Level Classes
-----------------

.. autoclass:: xgrammar.StructuralTag
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: StructuralTagItem
   :show-inheritance:
   :exclude-members: model_config

Format Union
------------

.. autodata:: Format

Basic Formats
-------------

.. autopydantic_model:: ConstStringFormat
   :show-inheritance:
   :exclude-members: model_config

.. autopydantic_model:: JSONSchemaFormat
   :show-inheritance:
   :exclude-members: model_config

.. autopydantic_model:: AnyTextFormat
   :show-inheritance:
   :exclude-members: model_config

.. autopydantic_model:: GrammarFormat
   :show-inheritance:
   :exclude-members: model_config

.. autopydantic_model:: RegexFormat
   :show-inheritance:
   :exclude-members: model_config

.. autopydantic_model:: QwenXMLParameterFormat
   :show-inheritance:
   :exclude-members: model_config

Combinatorial Formats
---------------------

.. autopydantic_model:: SequenceFormat
   :show-inheritance:
   :exclude-members: model_config

.. autopydantic_model:: OrFormat
   :show-inheritance:
   :exclude-members: model_config

.. autopydantic_model:: TagFormat
   :show-inheritance:
   :exclude-members: model_config

.. autopydantic_model:: TriggeredTagsFormat
   :show-inheritance:
   :exclude-members: model_config

.. autopydantic_model:: TagsWithSeparatorFormat
   :show-inheritance:
   :exclude-members: model_config
