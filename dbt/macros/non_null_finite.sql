{% macro non_null_finite(col) %}
SELECT '{{ col }}' AS column_name, {{ col }} AS value
FROM {{ ref('tour_averages') }}
WHERE {{ col }} IS NULL
   OR {{ col }} = 'NaN'::DOUBLE PRECISION
   OR {{ col }} = 'Infinity'::DOUBLE PRECISION
   OR {{ col }} = '-Infinity'::DOUBLE PRECISION
{% endmacro %}
