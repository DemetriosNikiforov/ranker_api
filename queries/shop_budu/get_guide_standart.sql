SELECT 
    gs.id, 
    gs.name, 
    gt1.type, 
    gt1.name AS gt_type_name,
    gt1.id as gt_type_id,
    gt1.parent_id AS parent_id,
    gt2.name AS parent_id_name
FROM guide_standard gs
LEFT JOIN guide_type gt1 ON gs.guide_type_id = gt1.id
LEFT JOIN guide_type gt2 ON gt1.parent_id = gt2.id
WHERE 
    gs.guide_type_id != 48 AND gs.guide_type_id != 123 AND
    gs.deleted_at IS NULL AND 
    gt1.deleted_at IS NULL
