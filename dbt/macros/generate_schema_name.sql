{# Use per-folder `+schema` values verbatim (silver/gold), falling back to the
   target schema from profiles.yml. The dbt default appends the custom schema
   to the target schema (gold + silver -> gold_silver), which is not the
   medallion layout this project declares. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name or target.schema }}
{%- endmacro %}
