SELECT
    gb."name" AS service_name,
    gs."name" as local_name,
    gb.local_id AS local_id,
    '{guide_base_id}' AS guide_base_id
FROM
    {guide_base_id} gb
    LEFT  JOIN guide_standard gs ON gb.local_id = gs.id
WHERE
    gb.deleted_at IS NULL

-- Path: queries/shop_budu/get_guide_base_by_id.sql