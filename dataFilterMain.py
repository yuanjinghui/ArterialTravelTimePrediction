import threading

import ROADWAY.roadwayFilter
import ROADWAY.highRiskFreq

t1 = threading.Thread(target=ROADWAY.roadwayFilter.roadwayFilter)
# t2 = threading.Thread(target=ROADWAY.highRiskFreq.highRiskFreq)

t1.start()
# t2.start()
#
# import PATM.patm
# import PATM.patmGenReportFiltered
#
#
# t3 = threading.Thread(target=PATM.patm.patm)
# t4 = threading.Thread(target=PATM.patmGenReportFiltered.patmGenReportFiltered)
#
# t3.start()
# t4.start()
#
# import BUS.bus
# import BUS.busHistory
#
# t5 = threading.Thread(target=BUS.bus.busFilter)
# t6 = threading.Thread(target=BUS.busHistory.butHistory)
#
# t5.start()
# t6.start()
