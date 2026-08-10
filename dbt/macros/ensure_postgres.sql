{# PostgreSQL-safe idempotent post-hook helpers.

dbt-postgres materializes `table` models by swapping a transient backup relation
into place (rename old -> backup, build new, drop backup) within the model
transaction. During that window the OLD table's constraint/index names still
exist in the schema, so a plain `ALTER TABLE ... ADD CONSTRAINT` or
`CREATE INDEX IF NOT EXISTS` on the freshly-swapped table either collides with
(stale backup) or is wrongly satisfied by (IF NOT EXISTS skipping against the
backup) the previous incarnation. `ADD CONSTRAINT` also has no `IF NOT EXISTS`.

These macros clean up the stale object by its name first, then recreate it on
the target relation, so they are safe on a first build, a plain rebuild, and a
--full-refresh. Constraint and index names are project-unique per model.
#}

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