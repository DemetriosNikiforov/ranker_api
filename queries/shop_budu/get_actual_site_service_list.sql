SELECT 
    gs.id, 
    gs.name, 
    gt1.type as global_type, 
    gt1.name AS gt_type_name,
    gt1.id AS guide_type_id,
    gt2.name AS parent_guide_type_name,
    gt1.parent_id AS parent_guide_type_id,
    gs.site_active as site_active,
	gs.sale_count as sale_count 
FROM guide_standard gs
inner JOIN guide_type gt1 ON gs.guide_type_id = gt1.id
inner JOIN guide_type gt2 ON gt1.parent_id = gt2.id
WHERE 
    gs.deleted_at IS NULL
    and gs.guide_type_id not in (123)
    AND gs.show_in_catalog = TRUE
    and gt1.active = true
    AND gt1.deleted_at IS NULL
ORDER BY gs.sale_count desc
