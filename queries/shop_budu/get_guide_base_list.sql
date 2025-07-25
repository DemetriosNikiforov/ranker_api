SELECT 
    id AS guide_base_id, 
    name AS lpu_name 
FROM guide_list gl 
WHERE deleted_at IS NULL
-- Path: queries/shop_budu/get_guide_base_list.sql

