import matplotlib.pyplot as plt
# mean_iou
# log = [0.13907290162771196, 0.21191489428174679, 0.23429238511652956, 0.2511331557116745, 0.2679265985201294, 0.2806143097083715, 0.2922029521327674, 0.30164146235268113, 0.31057133802835907, 0.3216531207539109, 0.3308739374745748, 0.33896323102041964, 0.347766589357604, 0.3561383921027092, 0.3644732450448859, 0.3721300072135735, 0.3803095955283834, 0.38914203499766314, 0.396902432177731, 0.40440615743167546, 0.4121606390890908, 0.41929834972752067, 0.42736168315787987, 0.43306426938181314, 0.4400248020939617, 0.44743927062880146, 0.4537870390667977, 0.4615267546930389, 0.4686043066128007, 0.474778181828177, 0.4813061836442879, 0.48692702246175784, 0.49327006917683147, 0.4985734192529361, 0.5044400073004702, 0.5087515962647273, 0.5139801731926443, 0.5196101718007127, 0.5240039198086778, 0.52918933591172, 0.5337391020348592, 0.5384509729214061, 0.5421642187791331, 0.5469051624405933, 0.5527164400778154, 0.5548850738546757, 0.5588418929571589, 0.5631563356743426, 0.5668310155654852, 0.571193221510653, 0.5752053753648018, 0.5787521288857748, 0.5817212352267181, 0.5840796307565667, 0.5849894120626493, 0.5880244392768958, 0.5883581676441209, 0.5907830766203436, 0.5927723048898154, 0.594492181151302, 0.596305054173038, 0.5975841974198058, 0.5995507178788121, 0.6008970880321002, 0.6018693954330734, 0.6036919389670514, 0.60600879126029, 0.6069094117156232, 0.6098933368462794, 0.6112061933628006, 0.6130155537472037, 0.6131360471560959, 0.6156602955620355, 0.6169234924578466, 0.6173359363519609, 0.6196517579900835, 0.6217243073490023, 0.6223549317289929, 0.6228363655904486, 0.6247855285244077, 0.6263108559392918, 0.6273619110079572, 0.6289805763508413, 0.6297334790989579, 0.6312475617184298, 0.6324904810690111, 0.6335715248855022, 0.6343556423827258, 0.6364722007861104, 0.6364342543725794, 0.6394625078627948, 0.6382310005229423, 0.6408036941931653, 0.6416120571377193, 0.6416279819432511, 0.6433224463334452, 0.6441512708625065, 0.6451777011556904, 0.6465916393287807, 0.6484686490544826]
#
# xiangjian = [0.06551559668281275, 0.1946891575761, 0.22814845098065478, 0.259179531848029, 0.282561103948141, 0.2986503445868734, 0.3096018068904175, 0.32026243222512674, 0.3289342959963649, 0.3358890212511588, 0.34346745099781273, 0.34990569436451713, 0.35849540473319214, 0.366250401464398, 0.373375289698774, 0.3803444576640902, 0.3893098878302206, 0.39461978917366264, 0.4012895075063225, 0.40829737930342447, 0.41570398184030277, 0.42107014966130224, 0.4281389981691754, 0.43273576435546823, 0.43568304244197764, 0.43725856203367774, 0.44076576756605534, 0.4439400976804791, 0.45053061768901753, 0.455689558899066, 0.46143088676535043, 0.4675387653583636, 0.47573254690535216, 0.4817703782547338, 0.4864437444207728, 0.4922905684590379, 0.496159149684283, 0.5045765848264967, 0.5080269958554408, 0.5136325165807335, 0.5197925909089423, 0.5234014477530684, 0.5298257013447513, 0.5343523434890066, 0.537796823092116, 0.5437149432445681, 0.5465918137224108, 0.5509902995444701, 0.5539197912776981, 0.5603909966007737, 0.565724065404926, 0.5688313159091302, 0.5710749723359545, 0.5724314444438183, 0.5741518329960742, 0.5755252253775506, 0.5780915109087009, 0.580796190768806, 0.5812998839654858, 0.5858854219992564, 0.5856648871821313, 0.5884324422630979, 0.5905878665256388, 0.5904706867230974, 0.5931210553856083, 0.5958587792364555, 0.5965450284024715, 0.5996470095081408, 0.6014590797256637, 0.6023049317140482, 0.604181616109565, 0.6047139836428441, 0.6065482505880481, 0.6098788629812323, 0.6121513736521653, 0.6115807349567063, 0.614448412650211, 0.6157971975835766, 0.6186431233977018, 0.6171810287377987, 0.6207004115397972, 0.6205705655899874, 0.6212994338277382, 0.6236129996940429, 0.6254670164102777, 0.6257843040789339, 0.6282384885958959, 0.6294495321165846, 0.6285277096005519, 0.6318227312999269, 0.633002751261343, 0.6341527759913242, 0.6352748326849607, 0.6363236660747595, 0.636417953127957, 0.6373634442781962, 0.639171313936593, 0.6395219312164521, 0.642272806535853, 0.6414280278522208]


# mean_acc
log = [0.2202499305905677, 0.3303257413513082, 0.375382857765416, 0.4081414014920696, 0.4355151586017586, 0.4579002668359238, 0.47578262535140436, 0.4937084741360403, 0.509702043424359, 0.5247468034559238, 0.5381732502834746, 0.5512518954915062, 0.5646912649362653, 0.5761155838418736, 0.5864658686702281, 0.596903854918262, 0.6054722965728286, 0.6154484901753555, 0.6235419136680247, 0.6313328350728816, 0.6392845554025611, 0.6457433170770471, 0.6525469801935083, 0.6587248716950652, 0.6648965730496079, 0.6698731372289258, 0.6757685537425434, 0.681092692608688, 0.6853267089237115, 0.6908067613760549, 0.6946167163806413, 0.6976668728035086, 0.7015684419981155, 0.7050972898878372, 0.7083992571760319, 0.7115314391048131, 0.7133480190825037, 0.7167497363952092, 0.7188954601222635, 0.7223308510244982, 0.7232491122738565, 0.7260831219933606, 0.7273933110227936, 0.7296996670498412, 0.7310922152052618, 0.7329699425971148, 0.7337887310458377, 0.7352460089277972, 0.7366746127412824, 0.7387948316203886, 0.7409534750469176, 0.7437088504577779, 0.7446740339426823, 0.7459583251230607, 0.7466625802762173, 0.7470844961555386, 0.7475508744237674, 0.7482434132629194, 0.748513632914274, 0.7490935004131106, 0.7501970401617569, 0.75028280078871, 0.751208448329149, 0.7517639741096144, 0.7525401387900639, 0.7525440085232601, 0.7536433136307543, 0.7540578301315038, 0.7537789967615525, 0.7548565453451186, 0.7553761756335787, 0.7561580789517723, 0.756571286940262, 0.7568187979657818, 0.757108768838372, 0.7574600025548213, 0.7573948389417646, 0.7586158684587758, 0.7587834779594723, 0.7592317475095424, 0.759152627870459, 0.759669127198612, 0.7597038205682264, 0.7609851612383973, 0.7602437206786867, 0.7614576073927035, 0.7616622253607613, 0.7613572145424199, 0.7625336855347334, 0.7618562326417584, 0.763011156917462, 0.7631175204768623, 0.762826682079935, 0.7641286703974549, 0.7632900380590878, 0.7644005850783946, 0.7645091142023523, 0.7641671399702883, 0.765202205275344, 0.7651048737110413]

xiangjian = [0.14788201000989223, 0.2881197488311804, 0.330075936878859, 0.37646198995250924, 0.4127950752823063, 0.4387021226938347, 0.4568019371105937, 0.47332421495111365, 0.488203224984099, 0.49972720013683597, 0.5131458548956349, 0.5234799234250084, 0.5346856815030289, 0.5455105345967446, 0.5557678547409628, 0.5639263704659615, 0.5722768772541664, 0.5809550627135108, 0.5876611433927877, 0.5953825621384361, 0.602255647229135, 0.6086180866957366, 0.6150407041568091, 0.6212764040909651, 0.6280360398560036, 0.6331632162492917, 0.6402874996456032, 0.64725924514172, 0.6533901079853798, 0.6592759125906532, 0.664283774780187, 0.6699372729716634, 0.6745464251033801, 0.6804930219118781, 0.6842347401665249, 0.688224195742618, 0.6926966967170127, 0.6965900623861506, 0.6990107705919516, 0.7022730668949697, 0.7054617803218222, 0.708846672976407, 0.7103524073376725, 0.7141738725454491, 0.7145784754437174, 0.7180393675544491, 0.7201584248561642, 0.7225044611271666, 0.722404544661713, 0.7244697770698638, 0.7277740157566769, 0.7309735543953961, 0.732556589728443, 0.7330458569272105, 0.7338290684350626, 0.7351256555302317, 0.7351552410340543, 0.7365211773480748, 0.7374500970883575, 0.7378715069697455, 0.7385782361448492, 0.7400095708030529, 0.7401012830243651, 0.7408305915488179, 0.7411893138048827, 0.7431825079394998, 0.7423698992575487, 0.7439684023701776, 0.7442840373736769, 0.7442582241375167, 0.7454519820348409, 0.7457292372160474, 0.7466826596642343, 0.7464908481852807, 0.7475460748694226, 0.7479600631339133, 0.7484239453343214, 0.7489447080828768, 0.7501853057919244, 0.7491889905109583, 0.7507120475424736, 0.7503676093923655, 0.7513684506955232, 0.7506546250459258, 0.7519143169301759, 0.7520176473952271, 0.7528830907019627, 0.7532249867581694, 0.7534701547831517, 0.7539688311311598, 0.7541938890822143, 0.7540787565114575, 0.7550324533989623, 0.7554364871344392, 0.7554490371408087, 0.7558136545621625, 0.755418513287707, 0.7571060694041994, 0.7565677522808124, 0.7563514948027769]


def plot(x,y,linestyle, label):
    plt.plot(x, y, linestyle, label=label)
    plt.legend()        # 显示上面的label


def ply(name, style):
    x_axis_data = [i for i in range(len(name[0]))]  # x

    plot(x=x_axis_data, y=name[0], linestyle=style, label='log(s)')
    plot(x=x_axis_data, y=name[1], linestyle=style, label='(g-s)')


    # plt.xlabel('Epoch Number')               # x_label
    # plt.ylabel('Mean IoU')

    plt.xlabel('Epoch Number')               # x_label
    plt.ylabel('Mean Class Accuracy')

    # plt.xticks(ticks=x_axis_data)
    # plt.yticks(ticks=[i for i in range(0, max)])      # 不显示所有y刻度标签

    # plt.grid(True)                  # 显示网格线

    # plt.ylim(-1,1)                #仅设置y轴坐标范围
    plt.show()


if __name__ == '__main__':
    list = [log, xiangjian]
    # i = 0
    # while i < len(list):
    #     list[i] = normalization_list(list[i])
    #     i = i + 1

    ply([name for name in list], style='-')
