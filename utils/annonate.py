import csv
gfile=open('test_labels.csv')
gtreader = csv.reader(gfile)

with open('new_test_labels.csv','a',newline='')as f:
    writer = csv.writer(f,dialect='excel')
    for row in gtreader:
        if row[3] == '0':
            row[3] = 'speed_limit_20'
        elif row[3] == '1':
            row[3] = 'speed_limit_30'
        elif row[3] == '2':
            row[3] = 'speed_limit_50'
        elif row[3] == '3':
            row[3] = 'speed_limit_60'
        elif row[3] == '4':
            row[3] = 'speed_limit_70'
        elif row[3] == '5':
            row[3] = 'speed_limit_80'
        elif row[3] == '6':
            row[3] = 'restriction_ends_80'
        elif row[3] == '7':
            row[3] = 'speed_limit_100'
        elif row[3] == '8':
            row[3] = 'speed_limit_120'
        elif row[3] == '9':
            row[3] = 'no_overtaking'
        elif row[3] == '10':
            row[3] = 'no_overtaking_trucks'
        elif row[3] == '11':
            row[3] = 'priority_at_next_intersection'
        elif row[3] == '12':
            row[3] = 'priority_road'
        elif row[3] == '13':
            row[3] = 'give_way'
        elif row[3] == '14':
            row[3] = 'stop'
        elif row[3] == '15':
            row[3] = 'no_traffic_both_ways'
        elif row[3] == '16':
            row[3] = 'no_trucks'
        elif row[3] == '17':
            row[3] = 'no_entry'
        elif row[3] == '18':
            row[3] = 'danger'
        elif row[3] == '19':
            row[3] = 'bend_left'
        elif row[3] == '20':
            row[3] = 'bend_right'
        elif row[3] == '21':
            row[3] = 'bend_danger'
        elif row[3] == '22':
            row[3] = 'uneven_road'
        elif row[3] == '23':
            row[3] = 'slippery_road'
        elif row[3] == '24':
            row[3] = 'road_narrows'
        elif row[3] == '25':
            row[3] = 'construction'
        elif row[3] == '26':
            row[3] = 'traffic_signal'
        elif row[3] == '27':
            row[3] = 'pedestrain_crossing'
        elif row[3] == '28':
            row[3] = 'school_crossing'
        elif row[3] == '29':
            row[3] = 'cycles_crossing'
        elif row[3] == '30':
            row[3] = 'snow'
        elif row[3] == '31':
            row[3] = 'animals_ahead'
        elif row[3] == '32':
            row[3] = 'restriction_ends'
        elif row[3] == '33':
            row[3] = 'go_right'
        elif row[3] == '34':
            row[3] = 'go_left'
        elif row[3] == '35':
            row[3] = 'go_straight'
        elif row[3] == '36':
            row[3] = 'go_right_or_straight'
        elif row[3] == '37':
            row[3] = 'go_left_or_straight'
        elif row[3] == '38':
            row[3] = 'keep_right'
        elif row[3] == '39':
            row[3] = 'keep_left'
        elif row[3] == '40':
            row[3] = 'roundabout'
        elif row[3] == '41':
            row[3] = 'restriction_ends_overtaking'
        elif row[3] == '42':
            row[3] = 'restriction_ends_overtaking_trucks'
        writer.writerow(row)
    gfile.close()