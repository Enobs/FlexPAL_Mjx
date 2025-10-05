import mujoco
model = mujoco.MjModel.from_xml_path('pickandplace.xml')
data  = mujoco.MjData(model)
print(model.sensor_adr)
mujoco.mj_step(model, data)
print(data.sensordata)