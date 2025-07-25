select gs.name as local_name, model_id as local_id, synonym  from synonym s 
left join guide_standard gs on gs.id = s.model_id 
where  model_type = '\App\Models\GuideStandard'
and s.deleted_at isnull