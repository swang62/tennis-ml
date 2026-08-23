
{% macro ensure_primary_key_sql(relation, conname, columns) -%}
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '{{ conname }}') THEN
        EXECUTE format(
            'ALTER TABLE %s DROP CONSTRAINT {{ conname }}',
            (SELECT conrelid::regclass::text
             FROM pg_constraint WHERE conname = '{{ conname }}' LIMIT 1)
        );
    END IF;
    ALTER TABLE {{ relation.schema }}.{{ relation.identifier }}
        ADD CONSTRAINT {{ conname }} PRIMARY KEY ({{ columns }});
END
$$;
{%- endmacro %}

{% macro ensure_not_null_sql(relation, columns) -%}
ALTER TABLE {{ relation.schema }}.{{ relation.identifier }}
{% for col in columns %}
    ALTER COLUMN {{ col }} SET NOT NULL{{ "," if not loop.last else ";" }}
{% endfor %}
{%- endmacro %}

{% macro ensure_index_sql(relation, idxname, columns) -%}
DO $$
DECLARE
    target regclass := '{{ relation.schema }}.{{ relation.identifier }}'::regclass;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_index ix JOIN pg_class i ON i.oid = ix.indexrelid
        WHERE i.relname = '{{ idxname }}' AND ix.indrelid = target
    ) THEN
        IF EXISTS (
            SELECT 1 FROM pg_index ix JOIN pg_class i ON i.oid = ix.indexrelid
            WHERE i.relname = '{{ idxname }}' AND ix.indrelid <> target
        ) THEN
            EXECUTE 'DROP INDEX ' || (
                SELECT i.oid::regclass::text FROM pg_index ix
                JOIN pg_class i ON i.oid = ix.indexrelid
                WHERE i.relname = '{{ idxname }}' AND ix.indrelid <> target
                LIMIT 1
            );
        END IF;
        EXECUTE format(
            'CREATE INDEX {{ idxname }} ON {{ relation.schema }}.{{ relation.identifier }} ({{ columns }})'
        );
    END IF;
END
$$;
{%- endmacro %}