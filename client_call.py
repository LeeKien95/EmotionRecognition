import requests

BASE_URL = 'http://10.8.75.135:8089'
landmarkChange = [7.8054411249516136, -0.71547500419853094, 7.8525024761978983, -0.6463011330943047, 7.6253081786376242, -0.65827245603927054, 7.6824029031348289, -0.64285143185077942, 7.3717605761647178, -0.41685887787173215, 7.0874954582861278, 0.14987740740923527, 6.4067780971288073, 0.16068612569601726, 6.20243080107997, 0.15393250476068943, 4.6388646644512335, 0.16402446325653841, 3.7114587186735264, -0.87401329773226166, 3.9244705724858306, -1.8340131847457144, 4.1737600749395938, -2.7005683975813781, 4.5507967450342051, -3.6038643868855615, 5.5137344127211065, -4.0067901984060939, 6.2947102767401191, -4.0707894841663688, 7.0551508031627037, -4.2564955985740198, 7.8111079345984251, -4.2848430714809638, 7.4361852136755076, -3.6980817725968222, 3.6876464582333952, -4.9805590968298645, 1.9039689363972343, -4.1199932418242469, -0.90987567025456428, -2.9332711032417791, -1.3112567766804304, -1.5953454159952685, 0.71566938079394049, -3.1698606408941714, 3.0568165367602091, -2.6490512315099011, 6.3245765264741607, -2.3007324892269594, 7.6206481325915689, -2.1906864427813275, 6.6199902988317731, -1.4338095447856745, 7.7701274638775502, -4.1616097708085249, 7.585648516683662, -4.070158827114426, 7.2997753563017511, -3.3478930837979988, 6.9795344813474571, -2.4201622222218759, 8.5597236115280566, -2.0550334405812549, 7.7477933884152606, -2.9200839169125459, 8.5942939084799264, -3.8876667741603228, 8.2422626724986117, -3.4743915060986126, 8.3740022685458655, -3.1667898435665904, 12.031395950864606, -0.9427904165384291, 11.797065618875919, -1.9726978727335194, 10.86780296516261, -2.6727172333602169, 9.3984454412438936, -2.07822118535546, 9.439059624378686, -1.157078410019011, 8.9837131582662124, -0.22288602306352345, 7.6492996242424738, -2.9050662591615151, 10.00878419403378, -2.7435048353925993, 10.758159946323616, -4.1410738481283431, 8.4618290316810203, -3.6390165443657452, 8.7684963887221272, -3.3537898947800784, 8.9912015849976541, -3.3124260522100428, 14.423215004213347, 2.3688697232636429, 17.060169333801326, 2.2667050343139437, 14.972926517382547, -1.1042087574738275, 11.42152878694057, -3.031439790942386, 14.115941955350536, -5.0962822898086131, 13.95873710161132, -7.3024354681837451, 11.521497304351044, -8.9230779550485124, 11.931260099323538, -6.4458430694772346, 9.3731166247281692, -5.856239463228377, 10.629675486737256, -4.0722829839284316, 10.438286579622201, -1.5925532807904403, 12.825635460075972, 0.17303554951242006, 10.594217521485888, 0.73254760889514614, 10.62819533596263, -0.90376548710601412, 11.366844380436106, -3.2470563503938763, 10.053812307787126, -5.981785055068741, 10.065695027878917, -7.452436014242295, 10.077586928457777, -5.946318076838395, 11.514719360647462, -3.0672831234983278, 10.560248887496357, -0.85927324594094046]
sending_data = {'landmarkChange': landmarkChange}

#response = requests.post("{}/predict".format(BASE_URL), json = sending_data)
response = requests.post("{}/predict".format(BASE_URL), json = sending_data)
response.json()
print(response.json())
