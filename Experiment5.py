import pygmo as pg


#training
f1_nadir = 1714.097014789188
f2_nadir = 8743.509257847385
hv_ref = 11590413.913937228

hyp = pg.hypervolume([[900.817653, 381.190921]])
HV = hyp.compute([f1_nadir, f2_nadir])
HVR = HV / hv_ref

print(HVR)

#testing
f1_nadir = 1223.784276627915
f2_nadir = 6733.094262137269
hv_ref = 5654119.54251087

hyp = pg.hypervolume([[853.9676665, 511.2354949]])
HV = hyp.compute([f1_nadir, f2_nadir])
HVR = HV / hv_ref

print(HVR)
