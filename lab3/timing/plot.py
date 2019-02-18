import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set()


def plot_ts():
  t_values=[[25.91896, 25.91896, 24.37615, 24.65796, 28.43499, 26.00193, 23.60821, 22.85290, 23.23699, 24.31011, 23.22698, 22.03202, 22.47906, 23.62418, 22.37105, 22.24493, 21.72589, 23.72789, 21.31486, 21.70801, 21.28887, 21.66390, 21.76404, 22.03703, 21.29102, 21.50297, 21.77215, 21.91997, 21.27910, 21.28696, 21.29292, 20.99609, 20.97702, 22.21513, 21.35801, 20.72716, 21.09909, 21.35897, 21.33393, 21.05999, 21.03806, 21.83700, 21.52586, 21.54613, 22.69197, 21.55995, 25.29192, 21.48700, 21.35110, 21.19088],
  [40.75909, 40.69901, 40.97009, 42.64402, 41.87894, 40.98511, 40.49397, 40.26079, 40.32683, 40.72118, 40.74812, 39.78705, 41.02015, 42.06109, 41.19182, 41.73207, 40.48109, 41.60595, 40.90905, 40.39001, 40.58695, 40.94410, 40.50088, 41.04304, 42.71293, 41.99386, 41.88609, 41.08095, 40.73501, 40.93909, 40.69018, 41.18800, 41.17990, 41.95094, 41.79692, 42.81783, 41.09406, 40.68613, 40.78317, 40.59386, 40.95602, 42.88697, 40.68708, 41.09693, 40.51280, 40.37094, 42.26995, 40.80296, 40.15803, 42.59610],
  [59.58080, 58.55393, 58.56299, 59.76796, 59.16405, 59.87906, 59.87096, 60.90593, 60.03022, 59.60608, 59.67212, 60.16588, 59.11493, 62.31999, 58.76803, 60.62102, 60.47511, 58.47883, 60.55188, 58.87389, 59.02290, 58.89416, 59.12995, 59.02004, 58.41804, 60.89783, 59.31401, 59.74388, 58.63094, 59.35311, 60.13703, 62.51597, 57.98984, 57.98793, 60.01210, 59.06391, 58.63404, 58.52699, 58.79498, 58.58898, 58.73609, 59.01098, 59.56316, 62.87813, 58.51316, 59.56507, 60.29701, 58.35509, 57.77812, 59.96490],
  [78.26495, 77.85201, 79.27799, 79.63181, 83.41503, 77.88086, 78.75395, 78.49216, 79.16403, 80.67894, 79.27489, 78.58515, 77.67797, 78.78494, 78.07517, 77.62623, 79.19598, 79.03385, 81.61497, 77.23308, 79.06890, 79.25487, 79.72693, 81.47001, 76.84493, 78.65191, 76.60484, 76.43414, 77.14200, 78.40395, 77.93713, 78.28593, 77.98314, 80.46913, 77.94094, 77.28195, 78.92299, 78.61710, 79.89812, 77.18110, 79.08702, 77.12197, 76.70188, 76.77007, 77.09002, 77.93689, 77.26598, 78.61090, 79.45919, 76.67589],
  [95.04700, 96.15088, 94.86103, 95.82806, 96.30799, 95.89601, 97.60404, 95.24107, 97.90707, 95.46995, 97.28885, 96.21882, 93.39190, 96.02594, 95.14785, 94.59996, 95.95299, 94.84100, 96.50993, 94.33699, 94.54799, 96.37904, 93.89520, 94.36107, 95.38388, 94.70320, 95.83092, 95.38794, 97.29314, 98.06585, 96.46416, 94.89393, 93.61601, 99.80822, 94.59901, 94.11907, 93.90402, 93.51993, 95.77298, 93.97602, 95.68906, 96.93694, 94.33389, 95.25800, 94.22302, 98.61398, 94.68293, 93.77408, 97.11385, 94.29097],
  [113.67893, 113.24811, 114.38012, 117.84410, 113.16991, 112.31804, 113.96194, 114.94112, 114.38990, 116.97912, 113.42502, 114.42614, 115.38196, 116.72807, 112.75196, 115.39483, 114.63594, 114.02202, 113.89208, 115.64183, 113.13391, 112.40292, 112.06794, 119.83609, 115.34405, 114.79878, 114.43901, 112.66804, 114.77113, 116.39905, 113.94596, 113.86395, 113.54804, 120.04399, 112.74910, 111.42302, 112.96606, 113.84487, 111.49311, 113.44790, 112.79202, 115.62586, 112.56790, 114.13717, 114.81905, 114.76111, 111.88793, 111.74202, 112.44512, 113.94000],
  [153.05400, 148.04292, 148.58699, 149.19090, 146.71302, 154.41704, 150.59519, 154.11782, 148.92507, 150.95210, 150.55609, 147.77994, 148.15497, 150.37704, 146.97599, 151.75605, 149.38498, 148.39387, 149.68109, 150.79403, 153.00608, 148.52405, 149.83487, 151.86501, 146.83199, 147.33005, 151.71099, 148.61512, 152.89807, 149.24502, 147.87197, 150.11287, 149.15395, 150.46215, 150.04611, 150.33007, 155.05219, 147.05491, 149.33705, 147.99595, 147.76707, 147.92490, 147.62688, 146.69585, 147.84408, 149.72305, 152.85993, 148.15187, 148.07606, 148.37003],
  [182.49202, 185.84704, 188.55500, 183.57491, 184.21102, 187.51907, 182.50990, 188.80010, 184.11899, 182.90496, 184.41582, 184.70192, 186.47814, 184.50499, 186.34796, 184.05008, 184.68809, 184.29589, 187.35886, 181.67400, 186.38301, 182.06096, 183.55584, 184.61084, 185.55307, 182.51395, 188.41910, 179.98815, 185.18710, 181.98395, 182.78909, 188.24911, 183.39920, 185.52518, 185.85205, 182.93118, 182.96790, 183.79784, 181.36811, 186.66196, 186.01918, 184.05294, 184.12995, 187.42085, 183.89988, 186.95092, 184.12304, 182.80983, 189.54992, 185.19521],
  [225.50583, 220.43419, 222.18013, 223.40488, 221.18783, 223.25397, 219.82002, 222.46408, 219.73896, 220.55888, 223.19889, 223.13285, 221.85397, 224.10512, 224.41006, 222.56708, 218.74714, 223.28806, 220.59894, 222.66793, 225.70181, 223.44708, 221.43602, 220.98684, 227.04697, 221.50302, 222.06712, 222.17607, 221.11797, 222.33701, 220.48092, 225.30103, 219.56086, 222.99218, 220.84594, 225.98314, 219.36607, 224.67899, 220.16597, 224.65515, 226.16982, 221.19498, 225.91805, 220.89791, 222.46408, 223.97113, 225.25692, 226.95303, 222.86701, 221.51613]]
  t_values = np.array(t_values)
  t_values[0][0] = t_values[0].mean()
  t_values = t_values.mean(axis=1)
  # print t_values

  Ts = [5,10,15,20,25,30,40,50,60]

  plt.plot(Ts, t_values, marker='o', label='Computation time')
  plt.title('MPPI timing vs. T parameter')
  plt.xlabel('T')
  plt.ylabel('Average time for MPPI call (ms)')
  legend = plt.legend(frameon=True)
  legend.get_frame().set_facecolor('white')
  legend.get_frame().set_edgecolor('black')
  legend.get_frame().set_alpha(0.7)
  plt.show()

def plot_ks():
  # k_values=[[532.45687, 84.93805, 86.48705, 81.31719, 79.46897, 80.25193, 82.78298, 80.17707, 80.48201, 80.26004, 80.48820, 80.77002, 81.11906, 79.95892, 80.23286, 81.03299, 82.21793, 81.39706, 80.42598, 79.52499, 81.45094, 79.23198, 79.47278, 79.58508, 79.16689, 78.92013, 79.73194, 79.15878, 79.57697, 79.43296, 82.10492, 78.89891, 78.29285, 79.14186, 79.77796, 81.86793, 79.89883, 80.70207, 78.91917, 77.98290, 78.09496, 80.58596, 79.68879, 78.99189, 80.43408, 78.24516, 79.06580, 79.31900, 82.83710, 78.40991],
  #           [78.43304, 78.05490, 78.02391, 79.00095, 80.48081, 79.14901, 77.64506, 77.75617, 78.28999, 79.02503, 79.17809, 78.73011, 77.58999, 77.98600, 78.41206, 78.74393, 79.50997, 78.32599, 82.46899, 78.42422, 78.14097, 78.09711, 79.22888, 81.01892, 76.64990, 78.19319, 77.84796, 77.79789, 78.99404, 77.81911, 78.20511, 77.83508, 79.51808, 77.08788, 78.94611, 78.02510, 77.85892, 80.30605, 78.46999, 76.88808, 79.82397, 77.78788, 77.47388, 78.77398, 79.45800, 79.37098, 79.78392, 78.23396, 77.88992, 77.54493],
  #           [78.07612, 77.21686, 76.82300, 77.30913, 78.36199, 78.90391, 77.75211, 79.11611, 77.68106, 77.11792, 78.05395, 78.25184, 80.61695, 77.95405, 77.22187, 77.67606, 77.93403, 77.30103, 78.76706, 78.04394, 77.87514, 77.85988, 80.43194, 77.11887, 77.81601, 80.30891, 78.52006, 77.67892, 78.10307, 77.97003, 77.26312, 78.03583, 79.05602, 77.06881, 78.97711, 78.18604, 80.43909, 78.34291, 77.74091, 79.29087, 79.11801, 78.81498, 77.23284, 79.69713, 77.52919, 78.76801, 79.44489, 77.32010, 76.95508, 78.13215],
  #           [81.55203, 82.48305, 78.84502, 79.85878, 81.72107, 80.12795, 79.72383, 81.41494, 80.06907, 80.75595, 84.60689, 81.25591, 81.29120, 81.25210, 80.34801, 82.73101, 82.16405, 81.53200, 80.22308, 82.76200, 80.28007, 81.79188, 80.27005, 82.21507, 83.69803, 78.75896, 79.64897, 81.75683, 79.93412, 81.23899, 80.53398, 80.34706, 80.21998, 82.37314, 81.34794, 80.71113, 80.08194, 81.24804, 84.89203, 81.63905, 80.73282, 79.81896, 78.51696, 82.36694, 81.93111, 80.08695, 80.76382, 80.52301, 80.09291, 80.99580],
  #           [93.45007, 92.92197, 96.48490, 106.42099, 95.59393, 94.10810, 94.66982, 95.43204, 93.96005, 94.47908, 96.39597, 93.85085, 94.36679, 93.29605, 95.69502, 93.54091, 93.01710, 94.60616, 93.25385, 94.08402, 93.87994, 93.39213, 89.72216, 78.62496, 77.60096, 78.93586, 76.90597, 79.09298, 78.90701, 78.60994, 79.10609, 78.08304, 77.57306, 79.07200, 78.75395, 78.10187, 77.90112, 77.49200, 76.92003, 77.93212, 78.90701, 78.65214, 76.96509, 77.54707, 76.98703, 78.38488, 78.33290, 77.71897, 77.32892, 78.59397],
  #           [125.91720, 95.23201, 78.51100, 78.46403, 79.53095, 78.93515, 78.31502, 78.74107, 79.72097, 79.03218, 78.54891, 76.94197, 78.17197, 78.28689, 78.20392, 79.22316, 79.98896, 79.74792, 81.48408, 81.44903, 80.67513, 81.36702, 80.67322, 81.04610, 80.15704, 89.36286, 90.54184, 91.47406, 91.15314, 91.73989, 90.79814, 90.13510, 92.52095, 91.57205, 91.76803, 90.61098, 90.39497, 91.68696, 90.41691, 90.16609, 90.21902, 91.36391, 90.05213, 90.43407, 90.81817, 91.27498, 90.46006, 89.91098, 91.13312, 92.17215],
  #           [130.42307, 82.96418, 83.26602, 82.46112, 82.89504, 85.86693, 83.94194, 82.83710, 83.95004, 89.07700, 91.01820, 90.66486, 90.34395, 96.19093, 100.97218, 100.88205, 102.34904, 101.03989, 100.83103, 100.92497, 102.76294, 101.93610, 100.92998, 100.84987, 107.77998, 129.62008, 133.97217, 134.63783, 136.39903, 135.40292, 137.88700, 136.13510, 134.71389, 120.10980, 102.94986, 101.70317, 101.46213, 102.57602, 100.43693, 100.42310, 101.81403, 101.83215, 101.40204, 99.99299, 100.36612, 102.64301, 100.33798, 101.37677, 101.72105, 100.15106],
  #           [214.10680, 151.41106, 106.13704, 100.36898, 100.79789, 100.82412, 100.21806, 100.46983, 104.12502, 110.33106, 112.53095, 112.53285, 129.83298, 132.36594, 136.42097, 137.36105, 136.59716, 140.08999, 167.29999, 175.94695, 182.19304, 179.02994, 179.47602, 179.17991, 181.97298, 178.91884, 181.15592, 178.28894, 179.05021, 178.61080, 182.13797, 179.10886, 178.90882, 180.93109, 178.72810, 178.63512, 178.56288, 180.87006, 178.08914, 181.45800, 178.74002, 179.16894, 178.35498, 181.55193, 179.55399, 180.61805, 178.86305, 178.79105, 178.82991, 178.66802],
  #           [267.58599, 244.74311, 248.69108, 279.08206, 320.30988, 317.41500, 317.14201, 317.47103, 318.88199, 318.03298, 320.74094, 318.59517, 318.83597, 318.62593, 320.99199, 319.24295, 318.46499, 318.41493, 319.38791, 280.56002, 248.54708, 248.50392, 251.16706, 248.58809, 224.13683, 205.68490, 204.51093, 207.02100, 204.63586, 207.00002, 205.31988, 205.30701, 204.91385, 196.61784, 184.80682, 188.13491, 186.79309, 185.51183, 174.89290, 174.33500, 162.75096, 162.00399, 165.67111, 157.99403, 156.99697, 156.26287, 156.53801, 157.42207, 155.82299, 158.73909],
  #           [339.78510, 262.87794, 248.31200, 253.74508, 253.37505, 259.41801, 285.89606, 296.23413, 291.80908, 296.61703, 292.66500, 281.73995, 292.80591, 291.75901, 288.83600, 295.20106, 293.63608, 294.06285, 255.93710, 249.15504, 242.85603, 242.59400, 240.31091, 233.59299, 229.22397, 224.56503, 224.85805, 227.04887, 225.15416, 222.12195, 202.84295, 202.88992, 202.98409, 202.80600, 200.14691, 202.07405, 200.12712, 200.19889, 199.85080, 200.82998, 201.32494, 199.91899, 199.65911, 199.56183, 202.58808, 199.53299, 203.47309, 201.07508, 200.26016, 200.23799],
  #           [378.12519, 382.86114, 397.77803, 397.56298, 398.94819, 394.27400, 393.48888, 392.45105, 380.11599, 380.07593, 391.42609, 392.93408, 393.96095, 390.95902, 392.47990, 390.80715, 390.56802, 393.19301, 393.01205, 391.28304, 392.65609, 383.76498, 390.49888, 392.16805, 394.46592, 390.92493, 393.17012, 374.55487, 381.10399, 393.62502, 394.87100, 391.29901, 391.49714, 391.95085, 389.84609, 392.09795, 394.05513, 392.25006, 394.36889, 393.79597, 390.83695, 391.12401, 393.79692, 390.30910, 390.98406, 393.06998, 392.12799, 395.62583, 393.78595, 390.74612],
  #           [723.40918, 701.12705, 696.31910, 697.61181, 700.14286, 698.99201, 694.90004, 698.16995, 697.99280, 696.20800, 699.88704, 696.25998, 694.27490, 698.63582, 696.21706, 697.94512, 695.36519, 698.53306, 702.70109, 695.36901, 699.11504, 697.25609, 698.03882, 699.90611, 699.12100, 696.74492, 695.09697, 698.35210, 697.74413, 695.21403, 696.96522, 698.28391, 698.13895, 699.87917, 697.39604, 694.90504, 696.38920, 697.92390, 696.96593, 697.02792, 697.35503, 697.09206, 697.89600, 698.33803, 697.07799, 698.72904, 695.93191, 698.22192, 701.85518, 699.14198]]
  k_values = [[924.83282, 101.42994, 96.82608, 98.65403, 96.10605, 92.73314, 97.51892, 90.84582, 86.63011, 90.21592, 87.10790, 87.23211, 84.32102, 85.02507, 85.69503, 86.27582, 84.47790, 84.22494, 86.01403, 86.51400, 84.95498, 83.21691, 86.16495, 85.29997, 86.05313, 83.91404, 86.44485, 85.50692, 86.60007, 84.09190, 84.97810, 84.96714, 85.52694, 85.74104, 85.91008, 84.43093, 83.87113, 84.78403, 86.40409, 88.01699, 84.79810, 85.66403, 85.10399, 85.87813, 84.40995, 86.01904, 82.06820, 83.34708, 85.10900, 86.82299],
  [87.47888, 87.29506, 84.20992, 88.16814, 84.96499, 84.36084, 84.61499, 87.53204, 85.46090, 84.37204, 85.60300, 80.98292, 86.86495, 84.14412, 87.03089, 83.96316, 86.46107, 85.34503, 86.93385, 81.21705, 85.23798, 89.48088, 85.56700, 86.69186, 85.06083, 87.93807, 83.06289, 85.73198, 84.16009, 85.30188, 86.02786, 86.27796, 84.58185, 84.99885, 86.30180, 85.99496, 85.07681, 84.60402, 87.28504, 84.89013, 81.27689, 83.31490, 85.93106, 84.73206, 83.37712, 83.33898, 83.82297, 84.99503, 83.59599, 82.91006],
  [73.79985, 73.89092, 83.61387, 75.97804, 81.52699, 87.58402, 84.47695, 85.79183, 85.37102, 83.31800, 83.93908, 82.91793, 72.95895, 73.08221, 72.14403, 74.84221, 71.72298, 71.78998, 73.13800, 72.09110, 76.90501, 72.02101, 71.63310, 71.30790, 72.84594, 71.44618, 72.47114, 71.78402, 71.96593, 72.44992, 71.91801, 76.76721, 72.64900, 73.16303, 71.43497, 71.60306, 73.01784, 71.22707, 71.77496, 71.73991, 70.90211, 70.44697, 73.47012, 69.77105, 70.13106, 71.82813, 74.39303, 69.56601, 70.60409, 69.30900],
  [73.15421, 73.10891, 73.73095, 73.82703, 76.05886, 73.16899, 72.74699, 72.96705, 78.07684, 73.14491, 73.28200, 73.30704, 73.11416, 77.25906, 72.83807, 72.80397, 73.06695, 74.15509, 73.83490, 72.93916, 73.39501, 72.76988, 72.12281, 77.14987, 73.29798, 72.24083, 74.50891, 73.98009, 75.49095, 73.52018, 75.40989, 73.09198, 72.87693, 72.70312, 71.50292, 74.84698, 77.85487, 72.52216, 76.98107, 73.96007, 73.14706, 72.49904, 74.84007, 77.78907, 74.90993, 75.96588, 74.05305, 73.73500, 73.10319, 76.16687],
  [89.86902, 88.51004, 88.95302, 81.36916, 71.58613, 69.07797, 71.65408, 69.50116, 69.18001, 70.66202, 70.79887, 70.73689, 70.46008, 71.16413, 69.91005, 71.22016, 70.13392, 68.86005, 70.31107, 71.15793, 71.29884, 69.84997, 69.49401, 69.98682, 71.24996, 69.73910, 70.42098, 68.12596, 69.08202, 69.82493, 71.85602, 69.99302, 71.51794, 70.91713, 70.80817, 79.22888, 88.33003, 88.94992, 87.26597, 91.32409, 89.22195, 90.02709, 88.45687, 89.66804, 89.62798, 87.03208, 87.37493, 87.28600, 90.81221, 87.87990],
  [123.95811, 131.11305, 99.65587, 71.92111, 68.93206, 68.85099, 69.36502, 70.44506, 70.01996, 69.91911, 67.98482, 69.78202, 69.55481, 71.06805, 69.66901, 69.68808, 68.65096, 69.86713, 71.98596, 70.05620, 71.66409, 69.50188, 69.30089, 70.69588, 72.06511, 72.29090, 72.73698, 72.77799, 71.82002, 72.90506, 72.18289, 71.91777, 72.43919, 73.62103, 82.93891, 84.37800, 83.56214, 83.51612, 85.72102, 85.09994, 84.09905, 83.63485, 85.16097, 84.40113, 84.92589, 84.01394, 84.09595, 84.71203, 83.76122, 83.40001],
  [219.53702, 82.48711, 73.61007, 72.99900, 72.21413, 72.65782, 78.25303, 83.96101, 82.13401, 82.87907, 82.86810, 93.89591, 96.39001, 97.01395, 97.57495, 96.84491, 97.47601, 97.58878, 98.55700, 125.31304, 130.27310, 136.81507, 133.21996, 131.64902, 130.54109, 132.87878, 132.94792, 134.94992, 131.53195, 133.50415, 135.09488, 132.82800, 133.25095, 132.01094, 131.99806, 132.38406, 134.44686, 133.52489, 130.88012, 134.09710, 131.60300, 134.59206, 131.18792, 133.80599, 136.74498, 132.78508, 132.14517, 133.92496, 132.36308, 131.06298],
  [198.75288, 108.57391, 103.26505, 99.62797, 106.90618, 113.60097, 112.65898, 112.03599, 112.27584, 111.91893, 112.48899, 126.10912, 130.57899, 135.49900, 161.89194, 176.28002, 178.61986, 177.85716, 180.81093, 178.25603, 181.15401, 178.18999, 178.46298, 179.93402, 177.76418, 179.45910, 177.84715, 180.49908, 177.46401, 178.10488, 177.97303, 168.02311, 143.77904, 143.84508, 139.63985, 135.88881, 136.43098, 135.06794, 136.45196, 137.36701, 136.23500, 135.64491, 138.20291, 136.13200, 135.89597, 135.96392, 135.39600, 137.36391, 137.23922, 134.79495],
  [425.10009, 370.47195, 185.59313, 201.13587, 203.78399, 203.79496, 203.82500, 244.29893, 264.71591, 315.01317, 314.56709, 317.80124, 317.73901, 318.02392, 311.30409, 247.88499, 250.69189, 247.42889, 246.62018, 249.63903, 246.28997, 224.32089, 219.67506, 220.10088, 221.88115, 221.58504, 221.87495, 220.13307, 219.70510, 221.56215, 206.11405, 201.00403, 184.37910, 175.04501, 161.67593, 161.38387, 163.56206, 161.25989, 163.37895, 162.14609, 162.91595, 162.79197, 161.83496, 156.46100, 146.70801, 147.34197, 147.58301, 146.85702, 146.41690, 146.40999],
  [248.86012, 248.24119, 250.93913, 265.39683, 283.77295, 291.75496, 293.35284, 301.93400, 311.49793, 292.32502, 291.51011, 283.37789, 251.51587, 240.35192, 239.16793, 240.44204, 223.40012, 219.11812, 231.16994, 229.54392, 229.44212, 230.47686, 222.62716, 219.74397, 216.10308, 216.65311, 217.01908, 199.73898, 201.38502, 196.65003, 198.30513, 197.09301, 199.00608, 199.27812, 199.04900, 197.00384, 197.74008, 197.11494, 199.64600, 197.20411, 199.48292, 197.20793, 197.88599, 197.92581, 199.33105, 197.62015, 199.46003, 198.05193, 197.44611, 197.26706],
  [370.50104, 383.38089, 403.90110, 402.04906, 396.91997, 392.92312, 391.09397, 387.71009, 378.07298, 389.92810, 388.89003, 389.87088, 391.49380, 386.47604, 389.50491, 390.23089, 388.24892, 391.95395, 385.89692, 384.41300, 389.68420, 390.95306, 388.55100, 391.12806, 390.00893, 387.50482, 389.81986, 389.62507, 387.92181, 391.54601, 389.85181, 387.41493, 386.08718, 389.54186, 387.67219, 390.07401, 390.16700, 388.15403, 387.38704, 389.96911, 388.10992, 390.25211, 390.28001, 388.78298, 389.04691, 382.56907, 378.77798, 386.66701, 385.38790, 389.98103],
  [714.36787, 695.27912, 703.33195, 692.60406, 694.48590, 694.52310, 693.75920, 693.03012, 693.56394, 694.20600, 693.72392, 691.11991, 692.94000, 694.78989, 692.38901, 694.67998, 692.29198, 690.97519, 695.15204, 690.58609, 691.29205, 691.84804, 692.99197, 692.62481, 691.94698, 693.74704, 692.53492, 696.59281, 691.48183, 694.80801, 693.42995, 692.43193, 691.10394, 693.92705, 690.75489, 692.86799, 689.21494, 692.36207, 692.61885, 690.98306, 692.66415, 692.45601, 693.01105, 691.46013, 692.75808, 692.98005, 694.25297, 693.24207, 691.90407, 692.33513],
  [2008.70705, 1294.26885, 1292.07706, 1289.49904, 1291.11195, 1289.93917, 1290.96889, 1290.87901, 1289.31403, 1291.54921, 1289.06989, 1291.08500, 1290.45606, 1291.29481, 1290.88998, 1290.36307, 1291.71801, 1290.20119, 1289.92105, 1290.10296, 1291.90898, 1289.35504, 1281.64601, 1290.33804, 1290.73191, 1290.26389, 1290.41314, 1290.41314, 1290.19308, 1289.93893, 1289.33096, 1291.68010, 1290.47012, 1290.39407, 1289.41298, 1291.52608, 1289.98709, 1289.31499, 1289.79206, 1290.14397, 1290.41100, 1289.67309, 1292.05585, 1289.78300, 1290.47680, 1289.52789, 1290.61604, 1288.43999, 1290.24696, 1289.34288],
  [1993.47401, 1906.31795, 1905.91407, 1905.57599, 1906.31700, 1905.65205, 1906.62193, 1905.50303, 1907.77493, 1905.92098, 1904.93393, 1906.82411, 1905.86305, 1905.45201, 1906.15892, 1904.10900, 1907.62591, 1905.63416, 1905.14302, 1906.69107, 1906.23903, 1903.83101, 1905.73406, 1904.19912, 1906.17299, 1905.00998, 1905.08890, 1906.38804, 1905.59506, 1906.66008, 1904.85811, 1904.38795, 1904.68001, 1904.64807, 1905.88617, 1903.74899, 1905.41816, 1906.96502, 1904.39105, 1908.20694, 1906.77094, 1905.15590, 1905.64108, 1903.79500, 1905.43604, 1904.53196, 1905.43914, 1906.05712, 1906.75712, 1905.13015],
  [2639.18185, 2520.83588, 2522.76015, 2522.52603, 2524.79696, 2522.84193, 2521.01207, 2524.24884, 2522.27998, 2520.36190, 2520.31302, 2520.47420, 2522.00198, 2522.52793, 2524.78886, 2523.41485, 2523.91505, 2522.04704, 2521.27719, 2521.38996, 2522.63594, 2521.77000, 2520.75195, 2521.39401, 2522.38989, 2522.33911, 2521.81911, 2520.13278, 2523.34785, 2521.79718, 2521.90518, 2523.40889, 2520.66588, 2522.11213, 2520.58101, 2519.95611, 2523.01693, 2522.08805, 2525.70915, 2523.03100, 2522.25304, 2523.60392, 2522.26901, 2523.76390, 2522.04680, 2523.86594, 2520.84303, 2525.02394, 2522.08996, 2522.07994],
  [6500.64993, 6282.73010, 6284.18398, 6286.09991, 6287.83512, 6287.56094, 6288.33008, 6284.25384, 6285.18605, 6285.30288, 6286.73100, 6285.84194, 6281.65197, 6288.02490, 6281.06403, 6288.11908, 6284.64293, 6283.92291, 6286.84592, 6283.80394, 6284.16800, 6286.01193, 6287.23192, 6287.98389, 6284.13510, 6287.53901, 6285.24399, 6284.60693, 6285.76708, 6289.26706, 6287.11796, 6284.40499, 6289.88004, 6283.05483, 6284.22284, 6289.92391, 6284.04284, 6286.19504, 6283.01501, 6281.56590, 6280.55501, 6286.92198, 6284.95097, 6284.06000, 6283.49304, 6281.96096, 6282.19986, 6283.89716, 6281.02303, 6283.82993]]

  k_values = np.array(k_values)
  k_values[0][0] = k_values[0].mean()
  k_values = k_values.mean(axis=1)
  print k_values

  Ks = np.array([100,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800, 409600,614400,819200,2048000])
  # k_values /= Ks # compute throughput

  plt.plot(Ks[:12], k_values[:12], marker='o', label='Computation time')
  plt.title('MPPI timing vs. K parameter')
  plt.xlabel('K')
  plt.ylabel('Average time for MPPI call (ms)')
  # plt.xscale('log')
  legend = plt.legend(frameon=True)
  legend.get_frame().set_facecolor('white')
  legend.get_frame().set_edgecolor('black')
  legend.get_frame().set_alpha(0.7)
  plt.show()

plot_ts()
plot_ks()