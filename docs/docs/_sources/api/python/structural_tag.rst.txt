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

.. autoclass:: ConstStringFormat
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: JSONSchemaFormat
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: AnyTextFormat
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: GrammarFormat
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: RegexFormat
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: QwenXMLParameterFormat
   :show-inheritance:
   :exclude-members: model_config

Combinatorial Formats
---------------------

.. autoclass:: SequenceFormat
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: OrFormat
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: TagFormat
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: TriggeredTagsFormat
   :show-inheritance:
   :exclude-members: model_config

.. autoclass:: TagsWithSeparatorFormat
   :show-inheritance:
   :exclude-members: model_config
