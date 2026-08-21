{# Idempotent post-hook helpers safe under dbt-postgres's table rebuild.

dbt-postgres swaps table models (rename old -> backup, build new, drop backup)
within one transaction, so the OLD constraint/index names still exist while the
new table is created: a plain ADD CONSTRAINT collides with the backup, and
CREATE INDEX IF NOT EXISTS skips against it. These macros drop the stale object
by name first, then recreate it on the target. Names are project-unique per
model. #}

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
{# NOT NULL is a per-column attribute, so there is no schema-global name to
collide with during dbt's table swap; re-setting it on a column that is already
NOT NULL is a no-op, which makes this idempotent without a catalog check.
Fails loudly when the model ever emits NULLs in a guaranteed column. #}
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