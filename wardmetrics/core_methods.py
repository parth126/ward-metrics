import numpy as np

def merge_events_if_necessary(events):
    index_to_remove = []

    for index in range(1, len(events)):
        if events[index - 1][1] == events[index][0]:
            events[index - 1] = (events[index-1][0], events[index][1])
            index_to_remove.append(index)

    index_to_remove.reverse()

    for index in index_to_remove:
        del events[index]

    return Event-based


def is_segment_in_interval(segment_start, segment_end, interval_start, interval_end):
    if interval_start <= segment_start and segment_end <= interval_end:
        return True
    return False


def get_standard_category_for_segment(segment_start, segment_end, ground_truth, detected_events):
    """
    Return standard category for a single segment
    :param segment_start:
    :param segment_end:
    :param ground_truth:
    :param detected_events:
    :return: "TP", "FP", "FN", "TN"
    """
    is_part_of_ground_truth = False
    is_part_of_detection = False

    for gt in ground_truth:
        if is_segment_in_interval(segment_start, segment_end, gt[0], gt[1]):
            is_part_of_ground_truth = True
            break

    for det in detected_events:
        if is_segment_in_interval(segment_start, segment_end, det[0], det[1]):
            is_part_of_detection = True
            break

    # decide which category
    if is_part_of_ground_truth:
        if is_part_of_detection:
            category = "TP"
        else:
            category = "FN"
    else:
        if is_part_of_detection:
            category = "FP"
        else:
            category = "TN"

    return category, is_part_of_ground_truth, is_part_of_detection


def get_segments_with_standard_error_categories(ground_truth_events, detected_events, evaluation_start=None, evaluation_end=None):
    segments = []

    gt_index = 0
    det_index = 0
    last_segment = None

    # TODO: handle events with zero length

    while True:
        if gt_index >= len(ground_truth_events):
            if det_index >= len(detected_events):
                if evaluation_end is not None:
                    category, is_gt, is_det = get_standard_category_for_segment(last_segment[1], evaluation_end, ground_truth_events, detected_events)
                    if is_gt:
                        gt_i = gt_index
                    else:
                        gt_i = -1
                    if is_det:
                        det_i = det_index
                    else:
                        det_i = -1
                    last_segment = (last_segment[1], evaluation_end, gt_i, det_i, category)
                    segments.append(last_segment)
                break
            gt_index = len(ground_truth_events) - 1
        elif det_index >= len(detected_events):
            det_index = len(detected_events) - 1

        gt_start = ground_truth_events[gt_index][0]
        gt_end = ground_truth_events[gt_index][1]
        d_start = detected_events[det_index][0]
        d_end = detected_events[det_index][1]

        if last_segment is not None:
            seg_start = last_segment[1]

            values_to_consider = []
            if gt_start > seg_start:
                values_to_consider.append(gt_start)
            if d_start > seg_start:
                values_to_consider.append(d_start)
            if d_end > seg_start:
                values_to_consider.append(d_end)
            if gt_end > seg_start:
                values_to_consider.append(gt_end)
            seg_end = min(values_to_consider)

            category, is_gt, is_det = get_standard_category_for_segment(seg_start, seg_end, ground_truth_events, detected_events)
            if is_gt:
                gt_i = gt_index
            else:
                gt_i = -1
            if is_det:
                det_i = det_index
            else:
                det_i = -1

            last_segment = (seg_start, seg_end, gt_i, det_i, category)
            segments.append(last_segment)
        else:
            # Create first segment:
            if evaluation_start is None:
                seg_start = min(gt_start, d_start)
                seg_end = min(max(gt_start, d_start), min(gt_end, d_end))
            else:
                seg_start = evaluation_start
                seg_end = min(gt_start, d_start)

            category, is_gt, is_det = get_standard_category_for_segment(seg_start, seg_end, ground_truth_events,
                                                                        detected_events)
            if is_gt:
                gt_i = gt_index
            else:
                gt_i = -1
            if is_det:
                det_i = det_index
            else:
                det_i = -1
            last_segment = (seg_start, seg_end, gt_i, det_i, category)
            segments.append(last_segment)

        if seg_end >= d_end:
            det_index += 1
        if seg_end >= gt_end:
            gt_index += 1

    index_to_remove = []
    for index, segment in enumerate(segments):
        if segment[1] - segment[0] <= 0:
            index_to_remove.append(index)

    index_to_remove.reverse()
    for index in index_to_remove:
        del segments[index]

    print("segments: ", segments)
    return segments


def score_segment(previous_segment, current_segment, next_segment):
    """
    Computing scores for current segment based on it's surroundings
    :param previous_segment: segment tuple for previous segment defined as (start, end, gt_event_index, det_event_index, standard_score) or None
    :param current_segment: segment tuple for current segment defined as (start, end, gt_event_index, det_event_index, standard_score)
    :param next_segment: segment tuple for next segment defined as (start, end, gt_event_index, det_event_index, standard_score) or None
    :return: return one of the following categories: 'TP' - true positive, 'TN' - true negative,
    'I' - insertion, 'M' - merge, 'D' - deletion, 'F' - fragmenting,
    'Os' - start overfill, 'Oe' - end overfill, 'Us' - start underfill, 'Ue' - end underfill,
    or 'no score' for errors
    """
    if current_segment is None:
        raise ValueError("current_segment must not be None.")
    index_of_standard_category = 4
    current_segment_score = "no score"

    if current_segment[index_of_standard_category] == "TP":
        current_segment_score = "TP"
    elif current_segment[index_of_standard_category] == "TN":
        current_segment_score = "TN"
    else:
        # Handle error categories:
        if previous_segment is not None and next_segment is not None:
            # normal case (in the middle):
            if current_segment[index_of_standard_category] == "FP":
                if (previous_segment[index_of_standard_category] == "TN" or previous_segment[index_of_standard_category] == "FN") and \
                        (next_segment[index_of_standard_category] == "TN" or next_segment[index_of_standard_category] == "FN"):
                    current_segment_score = "I"
                elif previous_segment[index_of_standard_category] == "TP" and next_segment[index_of_standard_category] == "TP":
                    current_segment_score = "M"
                elif (previous_segment[index_of_standard_category] == "TN" or previous_segment[index_of_standard_category] == "FN") and \
                                next_segment[index_of_standard_category] == "TP":
                    current_segment_score = "Os"
                elif previous_segment[index_of_standard_category] == "TP" and (
                        next_segment[index_of_standard_category] == "TN" or next_segment[index_of_standard_category] == "FN"):
                    current_segment_score = "Oe"
            elif current_segment[index_of_standard_category] == "FN":
                if (previous_segment[index_of_standard_category] == "TN" or previous_segment[index_of_standard_category] == "FP") and (
                        next_segment[index_of_standard_category] == "TN" or next_segment[index_of_standard_category] == "FP"):
                    current_segment_score = "D"
                elif previous_segment[index_of_standard_category] == "TP" and next_segment[index_of_standard_category] == "TP":
                    current_segment_score = "F"
                elif (previous_segment[index_of_standard_category] == "TN" or previous_segment[index_of_standard_category] == "FP") and \
                                next_segment[index_of_standard_category] == "TP":
                    current_segment_score = "Us"
                elif previous_segment[index_of_standard_category] == "TP" and (
                        next_segment[index_of_standard_category] == "TN" or next_segment[index_of_standard_category] == "FP"):
                    current_segment_score = "Ue"

        elif previous_segment is None and next_segment is not None:
            # start case (for the first segment):
            if current_segment[index_of_standard_category] == "FP":
                if next_segment[index_of_standard_category] == "TN" or next_segment[index_of_standard_category] == "FN":
                    current_segment_score = "I"
                elif next_segment[index_of_standard_category] == "TP":
                    current_segment_score = "Os"
            elif current_segment[index_of_standard_category] == "FN":
                if next_segment[index_of_standard_category] == "TN" or next_segment[index_of_standard_category] == "FP":
                    current_segment_score = "D"
                elif next_segment[index_of_standard_category] == "TP":
                    current_segment_score = "Us"

        elif previous_segment is not None and next_segment is None:
            # end case (for the last segment):
            if current_segment[index_of_standard_category] == "FP":
                if previous_segment[index_of_standard_category] == "TN" or previous_segment[index_of_standard_category] == "FN":
                    current_segment_score = "I"
                elif previous_segment[index_of_standard_category] == "TP":
                    current_segment_score = "Oe"
            elif current_segment[index_of_standard_category] == "FN":
                if previous_segment[index_of_standard_category] == "TN" or previous_segment[index_of_standard_category] == "FP":
                    current_segment_score = "D"
                elif previous_segment[index_of_standard_category] == "TP":
                    current_segment_score = "Ue"

        elif previous_segment is None and next_segment is None:
            # if only one segment is given (exceptional case):
            if current_segment[index_of_standard_category] == "FP":
                current_segment_score = "I"
            elif current_segment[index_of_standard_category] == "FN":
                current_segment_score = "D"

    return current_segment_score


def compute_detailed_segment_scores(segments):
    new_segments = []

    # Handle special case if only one segment exists:
    if len(segments) == 1:
        seg_score = score_segment(None, segments[0], None)
        n_seg = segments[0] + (seg_score,)  # Create new tuple (append to the end)
        new_segments.append(n_seg)
        return new_segments

    # handle first segment:
    seg_score = score_segment(None, segments[0], segments[1])
    n_seg = segments[0] + (seg_score,) # Create new tuple (append to the end)
    new_segments.append(n_seg)

    # Handle segments in the middle:
    for i in range(1, len(segments) - 1):
        seg_score = score_segment(segments[i-1], segments[i], segments[i+1])
        n_seg = segments[i] + (seg_score,)  # Create new tuple (append to the end)
        new_segments.append(n_seg)

    # handle last segment:
    seg_score = score_segment(segments[-2], segments[-1], None)
    n_seg = segments[-1] + (seg_score,)  # Create new tuple (append to the end)
    new_segments.append(n_seg)

    return new_segments


def count_segment_scores(segments):
    categories = ["TP", "TN", "I", "D", "F", "M", "Os", "Oe", "Us", "Ue"]
    results = {}

    # Init values:
    for c in categories:
        results[c] = 0

    # Calculate total segment length for each category
    for s in segments:
        results[s[5]] += s[1] - s[0]

    # Calculate normed values:
    eval_length = segments[-1][1] - segments[0][0]
    results_normed = {}
    for c in categories:
        results_normed[c] = results[c]/eval_length

    return results, results_normed


def twoset_metrics(segment_counts):
    P = segment_counts["D"] + segment_counts["F"] + segment_counts["Us"] + segment_counts["Ue"] + segment_counts["TP"]
    N = segment_counts["I"] + segment_counts["M"] + segment_counts["Os"] + segment_counts["Oe"] + segment_counts["TN"]

    dr = segment_counts["D"]/P
    fr = segment_counts["F"]/P
    us = segment_counts["Us"] / P
    ue = segment_counts["Ue"] / P
    tpr = 1 - (dr + fr + us + ue)

    ir = segment_counts["I"]/N
    mr = segment_counts["M"] / N
    o_s = segment_counts["Os"] / N
    oe = segment_counts["Oe"] / N
    fpr = ir + mr + o_s + oe

    results = {"dr": dr, "fr": fr, "us": us, "ue": ue, "tpr": tpr,
               "ir": ir, "mr": mr, "os": o_s, "oe": oe, "fpr": fpr }
    return results


def _get_ground_truth_event_index_list(segments):
    index_list = []
    for s in segments:
        if s[2] not in index_list and s[2] != -1:
            index_list.append(s[2])
    return index_list


def _get_detected_event_index_list(segments):
    index_list = []
    for s in segments:
        if s[3] not in index_list and s[3] != -1:
            index_list.append(s[3])
    return index_list


def _get_segments_for_ground_truth_event(segments, event_index):
    return [s for s in segments if s[2] == event_index]


def _get_segments_for_detected_event(segments, event_index):
    return [s for s in segments if s[3] == event_index]


def _score_ground_truth_event(segments_for_event):
    # get segment score values as a list:
    segment_scores = []
    for s in segments_for_event:
        segment_scores.append(s[5])

    # get event's score:
    current_event_score = ""
    if segment_scores.count("TP") == 1:
        current_event_score += "C"
    elif len(segment_scores) == 1 and segment_scores.count("D") == 1:
        current_event_score += "D"

    if segment_scores.count("F") > 0:
        current_event_score += "F"

    return current_event_score


def _score_detected_event(segments_for_event):
    # get segment score values as a list:
    segment_scores = []
    for s in segments_for_event:
        segment_scores.append(s[5])

    # get event's score:
    current_event_score = ""
    if segment_scores.count("TP") == 1:
        current_event_score += "C"
    elif len(segment_scores) == 1 and segment_scores.count("I") == 1:
        current_event_score += "I'"

    if segment_scores.count("M") > 0:
        current_event_score += "M'"

    return current_event_score


def _have_overlapping_segments(segments_1, segments_2):
    overlap = False
    for s in segments_1:
        if s in segments_2:
            overlap = True
            break
    return overlap


def compute_event_scores(segments):
    # Get list of events indexes:
    detected_indexes = _get_detected_event_index_list(segments)
    ground_truth_indexes = _get_ground_truth_event_index_list(segments)
    print("DET: ", detected_indexes)
    print("GT: ", ground_truth_indexes)
    # get score for each gt event:
    gt_event_scores = []
    for i in ground_truth_indexes:
        current_segments = _get_segments_for_ground_truth_event(segments, i)
        e_score = _score_ground_truth_event(current_segments)
        gt_event_scores.append(e_score)

    # get score for each detection event
    det_event_scores = []
    for i in detected_indexes:
        current_segments = _get_segments_for_detected_event(segments, i)
        e_score = _score_detected_event(current_segments)
        det_event_scores.append(e_score)

    # cross check event scores for merging and fragmented results:
    for i in ground_truth_indexes:
        for j in detected_indexes:
            segments_gt = _get_segments_for_ground_truth_event(segments, i)
            segments_det = _get_segments_for_detected_event(segments, j)
            if _have_overlapping_segments(segments_gt, segments_det):
                # change ground truth event score if needed:
                if det_event_scores[j] == "M'" or det_event_scores[j] == "M'F'":
                    gt_event_scores[i] += "M"

                # change detected event score if needed:
                if gt_event_scores[i] == "F" or gt_event_scores[i] == "FM":
                    det_event_scores[j] += "F'"

    # clean up event score labels:
    for i in range(len(gt_event_scores)):
        if "F" in gt_event_scores[i] and "M" in gt_event_scores[i]:
            gt_event_scores[i] = "FM"
        elif "C" in gt_event_scores[i] and "M" in gt_event_scores[i]:
            gt_event_scores[i] = "M"
    for i in range(len(det_event_scores)):
        if "F" in det_event_scores[i] and "M" in det_event_scores[i]:
            det_event_scores[i] = "FM'"
        if "C" in det_event_scores[i] and "F" in det_event_scores[i]:
            det_event_scores[i] = "F'"

    return gt_event_scores, det_event_scores


def _count_event_scores(gt_event_scores, detection_scores):
    results = {
        "total_gt": len(gt_event_scores),
        "total_det": len(detection_scores),
        "D": gt_event_scores.count("D"),
        "F": gt_event_scores.count("F"),
        "FM": gt_event_scores.count("FM"),
        "M": gt_event_scores.count("M"),
        "C": gt_event_scores.count("C"),
        "M'": detection_scores.count("M'"),
        "FM'": detection_scores.count("FM'"),
        "F'": detection_scores.count("F'"),
        "I'": detection_scores.count("I'")
    }
    return results


def _get_detailed_event_metrics(segments):
    gt_event_scores, det_event_scores = compute_event_scores(segments)
    detailed_event_metrics = _count_event_scores(gt_event_scores, det_event_scores)
    return gt_event_scores, det_event_scores, detailed_event_metrics


def _get_standard_event_metrics(ground_truth_events, detected_events, gt_event_scores, detected_event_scores):
    # Compute recall:
    tp_gt = 0
    tp_gt_w = 0
    fn_gt = 0
    fn_gt_w = 0
    for i in range(len(ground_truth_events)):
        if gt_event_scores[i] == 'D':
            fn_gt += 1
            fn_gt_w += ground_truth_events[i][1] - ground_truth_events[i][0]
        else:
            tp_gt += 1
            tp_gt_w += ground_truth_events[i][1] - ground_truth_events[i][0]

    recall = tp_gt / (tp_gt + fn_gt)
    recall_w = tp_gt_w / (tp_gt_w + fn_gt_w)

    # Compute precision:
    tp_det = 0
    tp_det_w = 0
    fp_det = 0
    fp_det_w = 0
    for i in range(len(detected_events)):
        if detected_event_scores[i] == "I'":
            fp_det += 1
            fp_det_w += detected_events[i][1] - detected_events[i][0]
        else:
            tp_det += 1
            tp_det_w += detected_events[i][1] - detected_events[i][0]

    precision =  tp_det / (tp_det + fp_det)
    precision_w =  tp_det_w / (tp_det_w + fp_det_w)

    standard_metrics = {
        "precision": precision,
        "recall": recall,
        "precision (weighted)": precision_w,
        "recall (weighted)": recall_w
    }
    return standard_metrics


def eval_segments(ground_truth_events, detected_events, evaluation_start=None, evaluation_end=None):
    """
    Segment-based evaluation (frame - length based)

    Computes and scores segments and returns the occurrences of each error type in the overall dataset segments

    Args
    ----
        ground_truth_events: list of tuples (start, end) or lists [start, end]
            numeric values (e.g. frame number or posix timestamp) for ground truth events' start and end times
        detected_events: list of tuples (start, end) or lists [start, end]
            numeric values (e.g. frame number or posix timestamp) for detected events' start and end times
        evaluation_start: numeric value or None
            This should be the first segment's start value. None indicates that start of the first event should be used.
        evaluation_end: numeric value or None
            This should be the first segment's start value. None indicates that start of the first event should be used.

    Returns
    -------
        twoset_results: dictionary
            result for the 2SET metrics as a dictonary
        segments_with_detailed_categories: list of tuples
            list of detected segments including standard and detailed score categories
        segment_counts: dictionary
            frame counts/length of segments for each category
        normed_segment_counts: dictionary
            same as before but normed
    """
    if len(ground_truth_events) <= 0 or len(detected_events) <= 0:
        raise AttributeError("Insufficient data. List of ground truth or detected events is empty - calculation not possible.")

    ground_truth_events = merge_events_if_necessary(ground_truth_events)
    print("Updated GT: ", ground_truth_events)
    detected_events = merge_events_if_necessary(detected_events)
    print("Updated DT: ", detected_events)

    segments_with_category = get_segments_with_standard_error_categories(ground_truth_events, detected_events,
                                                                         evaluation_start, evaluation_end)
    segments_with_detailed_categories = compute_detailed_segment_scores(segments_with_category)

    segment_counts, normed_segment_counts = count_segment_scores(segments_with_detailed_categories)
    twoset_results = twoset_metrics(segment_counts)

    return twoset_results, segments_with_detailed_categories, segment_counts, normed_segment_counts


def eval_events(ground_truth_events, detected_events, evaluation_start=None, evaluation_end=None):
    """
    Event-based evaluation

    Assigns scores to each ground truth and detection event and calculates statistics

    Args
    ----
        ground_truth_events: list of tuples (start, end) or lists [start, end]
            numeric values (e.g. frame number or posix timestamp) for ground truth events' start and end times
        detected_events: list of tuples (start, end) or lists [start, end]
            numeric values (e.g. frame number or posix timestamp) for detected events' start and end times
        evaluation_start: numeric value or None
            This should be the first segment's start value. None indicates that start of the first event should be used.
        evaluation_end: numeric value or None
            This should be the first segment's start value. None indicates that start of the first event should be used.

    Returns
    -------
        gt_scores: list
            score label for each ground truth event
        detection_scores: list
            score label for each detected event
        detailed_score_statistics: dictionary
            containing total number of events for each score category
        standard_score_statistics: dictionary
            precision and recall values (normal and weighted with event length) based on standard event scores (TP, FP, TN, FN)
    """
    if len(ground_truth_events) <= 0 or len(detected_events) <= 0:
        raise AttributeError("Insufficient data. List of ground truth or detected events is empty - calculation not possible.")

    #ground_truth_events = merge_events_if_necessary(ground_truth_events)
    print("Updated GT: ", ground_truth_events)
    #detected_events = merge_events_if_necessary(detected_events)
    print("Updated DT: ", detected_events)
    segments_with_category = get_segments_with_standard_error_categories(ground_truth_events, detected_events, evaluation_start, evaluation_end)
    segments_with_detailed_categories = compute_detailed_segment_scores(segments_with_category)

    gt_scores, detection_scores, detailed_score_statistics = _get_detailed_event_metrics_new(segments_with_detailed_categories, ground_truth_events, detected_events)
    #gt_scores1, detection_scores1, detailed_score_statistics1 = _get_detailed_event_metrics(segments_with_detailed_categories)
    standard_score_statistics = _get_standard_event_metrics(ground_truth_events, detected_events, gt_scores, detection_scores)

    #return (gt_scores, gt_scores1), (detection_scores, detection_scores1), (detailed_score_statistics, detailed_score_statistics1), standard_score_statistics
    return gt_scores, detection_scores, detailed_score_statistics, standard_score_statistics

def _get_detailed_event_metrics_new(segments_with_category, gt_events, det_events):
    gt_event_scores, gt_seg_lengths, det_event_scores, det_seg_lengths = compute_event_scores_smart(gt_events, det_events)
    detailed_event_metrics = _count_event_scores_new(np.array(gt_event_scores), np.array(gt_seg_lengths), np.array(det_event_scores), np.array(det_seg_lengths))
    return gt_event_scores, det_event_scores, detailed_event_metrics

def compute_event_scores_new(gt_events, det_events):
    gt_category = []
    gt_lengths = []
    for s in gt_events:
        os_gt = find_overlapping_segments((s[0], s[1]), det_events)
        print(os_gt)
        if len(os_gt) == 1 and (os_gt[0][0], os_gt[0][1]) == (s[0], s[1]):
            gt_temp = ["C"]
            gt_temp_length = 1
        else:
            gt_temp = []
            gt_temp_length = s[1] - s[0]
            for s1 in os_gt:
                if (s1[0] <= s[0]) and (s1[1] >= s[1]): # s1 consumes s
                    gt_temp += ["M"]
                elif (s1[0] < s[0]) and (s1[1] < s[1]): # left of s overlaps right of s1
                    gt_temp += ["FM"]
                elif (s[0] <= s1[0]) and (s[1] >= s1[1]): # s consumes s1
                    gt_temp += ["F"]
                elif (s[0] < s1[0]) and (s[1] < s1[1]): # left of s1 overlaps right of s
                    gt_temp += ["FM"]
        print(gt_temp)
        if len(set(gt_temp)) > 1:
            gt_category.append("FM")
        else:
            gt_category.append(list(set(gt_temp))[0])
        gt_lengths.append(gt_temp_length)

    print(gt_category)
    det_category = []
    det_lengths = []
    for s in det_events:
        os_det = find_overlapping_segments((s[0], s[1]), gt_events)
        print(os_det)
        if len(os_det) == 1 and (os_det[0][0], os_det[0][1]) == (s[0], s[1]):
            det_temp = ["C"]
            det_temp_length = 1
        else:
            det_temp = []
            det_temp_length = s[1]-s[0]
            for s1 in os_det:
                if (s1[0] <= s[0]) and (s1[1] >= s[1]): # s1 consumes s
                    det_temp += ["F'"]
                elif (s1[0] < s[0]) and (s1[1] < s[1]): # left of s overlaps right of s1
                    det_temp += ["FM'"]
                elif (s[0] <= s1[0]) and (s[1] >= s1[1]): # s consumes s1
                    det_temp += ["M'"]
                elif (s[0] < s1[0]) and (s[1] < s1[1]): # left of s1 overlaps right of s
                    det_temp += ["FM'"]

        if len(set(det_temp)) > 1:
            det_category.append("FM'")
        else:
            det_category.append(list(set(det_temp))[0])
        det_lengths.append(det_temp_length)

    return gt_category, gt_lengths, det_category, det_lengths

def has_overlap(seg1, seg2):
    if (seg1[0] < seg2[1] and seg1[1] > seg2[0]):
        return True

def find_overlapping_segments(segment, all_segments):
    overlapping_segs = []
    for s in all_segments:
        if has_overlap(segment, s):
            overlapping_segs.append(s)
    return(overlapping_segs)

def _count_event_scores_new(gt_event_scores, gt_seg_lengths, detection_scores, det_seg_lengths):
    print("GT: ", gt_seg_lengths, gt_event_scores)
    print("GT: ", det_seg_lengths, detection_scores)
    results = {
        "total_gt": sum(gt_seg_lengths),
        "total_det": sum(det_seg_lengths),
        "D": sum(gt_seg_lengths[gt_event_scores == "D"]),
        "F": sum(gt_seg_lengths[gt_event_scores == "F"]),
        "FM": sum(gt_seg_lengths[gt_event_scores == "FM"]),
        "M": sum(gt_seg_lengths[gt_event_scores == "M"]),
        "C": sum(gt_seg_lengths[gt_event_scores == "C"]),
        "M'": sum(det_seg_lengths[detection_scores == "M'"]),
        "FM'": sum(det_seg_lengths[detection_scores == "FM'"]),
        "F'": sum(det_seg_lengths[detection_scores == "F'"]),
        "I'": sum(det_seg_lengths[detection_scores == "I'"])
    }
    print("RRR: ", results)
    return results


def create_adjacency_matrix(gt_event_dict, det_event_dict):

    adjacency_matix = np.zeros([len(gt_event_dict), len(det_event_dict)])

    for gt_event in gt_event_dict:

        gt_event_id = gt_event_dict[gt_event]
        overlapping_segments = find_overlapping_segments((gt_event[0], gt_event[1]), det_event_dict.keys())

        for os in overlapping_segments:
            det_event_id = det_event_dict[os]
            adjacency_matix[gt_event_id, det_event_id] = min(os[1], gt_event[1]) - max(os[0], gt_event[0])

    return adjacency_matix

def compute_event_scores_smart(gt_events, det_events):

    gt_event_dict = {}
    det_event_dict = {}
    gt_event_inv_dict = {}
    det_event_inv_dict = {}

    for g_ev in gt_events:
        gt_event_dict[g_ev] = len(gt_event_dict)
        gt_event_inv_dict[len(gt_event_dict) - 1] = g_ev

    for d_ev in det_events:
        det_event_dict[d_ev] = len(det_event_dict)
        det_event_inv_dict[len(det_event_dict) - 1] = d_ev

    adjacency_matix = create_adjacency_matrix(gt_event_dict, det_event_dict)

    valid_paths = getValidPathsLatest(adjacency_matix)

    best_alignment = get_best_alignment(valid_paths)
    print("Best: ", best_alignment, len(gt_event_dict))
    complete_alignment = fill_missing(best_alignment, len(gt_event_dict))
    print(complete_alignment)
    gt_category = []
    gt_lengths = []

    det_category = []
    det_lengths = []

    for segment in complete_alignment:
        if segment[1] == -1:
            gt_category.append("D")
            g_ev = gt_event_inv_dict[segment[0]]
            gt_lengths.append(g_ev[1]-g_ev[0])

        elif segment[0] == -1:
            det_category.append("I")
            g_ev = gt_event_inv_dict[segment[1]]
            det_lengths.append(d_ev[1] - d_ev[0])

        else:
            print(segment)
            g_ev = gt_event_inv_dict[segment[0]]
            d_ev = det_event_inv_dict[segment[1]]
            gt_lengths.append(g_ev[1]-g_ev[0])
            det_lengths.append(d_ev[1] - d_ev[0])
            if g_ev == d_ev:
                gt_category.append("C")
                det_category.append("C")
            elif (g_ev[0] <= d_ev[0]) and (g_ev[1] >= d_ev[1]):
                gt_category.append("F")
                det_category.append("F'")
            elif (d_ev[0] <= g_ev[0]) and (d_ev[1] >= g_ev[1]):
                gt_category.append("M")
                det_category.append("M'")
            else:
                print("FM: ", g_ev, d_ev)
                gt_category.append("FM")
                det_category.append("FM'")

    return gt_category, gt_lengths, det_category, det_lengths


def getValidPathsLatest(n, xshift = 0, yshift = 0):
    valid_paths = []
    if n.shape[1] == 1:
        for i in range(0,n.shape[0]):
            if n[i,0] > 0:
                valid_paths.append([(xshift+i, yshift, n[i,0])])
        valid_paths.append((-1, yshift, 0))
        return valid_paths

    new_path = [(-1, yshift, 0)]
    for p in getValidPathsLatest(n[:, 1:], xshift=xshift, yshift=yshift+1):
        if p:
            new_path.append(p)
    valid_paths.append(new_path)

    for i in range(0,n.shape[0]):
        if n[i,0] > 0:
            new_path = [(xshift+i, yshift, n[i,0])]
            for p in getValidPathsLatest(n[i+1:, 1:], xshift=xshift+i+1, yshift=yshift+1):
                if p:
                    new_path.append(p)
            valid_paths.append(new_path)
    return valid_paths


def flattenPathTree(g): # g == paths from a specific starting point
    flattened_paths = []
    if not isinstance(g, list):
        return [[g]]
    elif len(g) == 1:
        return([g])
    else:
        for i in range(1,len(g)):
            for p in flattenPathTree(g[i]):
                fp = [g[0]]
                fp += p
                flattened_paths.append(fp)
        return flattened_paths


def get_best_alignment(valid_paths):
    all_paths = []
    for i in range(0,len(valid_paths)):
        all_paths += flattenPathTree(valid_paths[i])
    max_score = 0.0
    for a in all_paths:
        if sum([x[2] for x in a]) > max_score:
            max_score = sum([x[2] for x in a])
            best_alignment = a

    return best_alignment

def fill_missing(path, max_x):
    missing_x = [x for x in range(0,max_x)]
    for p in path:
        if p[0] != -1:
            #print(missing_x)
            missing_x.remove(p[0])
    new_path = []
    current_missing = 0
    next_valid = 0
    while len(new_path) < len(path) + len(missing_x):
        if next_valid < len(path):
            p = path[next_valid]

            if (current_missing < len(missing_x)) and (p[0] > missing_x[current_missing]):
                new_path.append((missing_x[current_missing],-1,0))
                current_missing += 1
            else:
                new_path.append(p)
                next_valid += 1

        else:
            new_path.append((missing_x[current_missing],-1,0))
            current_missing += 1

    return new_path

