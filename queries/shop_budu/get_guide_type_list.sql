WITH RECURSIVE tree AS (
    -- Базовый случай: выбираем корневые элементы (элементы без parent_id)
    SELECT id AS main_id, name AS main_name, id AS child_id, name AS child_name, 0 AS level
    FROM guide_type
    WHERE parent_id IS NULL AND deleted_at IS NULL

    UNION ALL

    -- Рекурсивный случай: добавляем дочерние элементы
    SELECT tree.main_id, tree.main_name, gt.id AS child_id, gt.name AS child_name, tree.level + 1 AS level
    FROM guide_type gt
    INNER JOIN tree ON gt.parent_id = tree.child_id
    WHERE gt.deleted_at IS NULL
)
SELECT main_name, main_id, child_name, child_id
FROM tree
ORDER BY main_id, level;
