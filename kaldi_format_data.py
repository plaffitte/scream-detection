import numpy
import struct
# from fileutils import smart_open

def readPfile(filename):
    """
    Reads the contents of a pfile. Returns a tuple (feature, label), where
    both elements are lists of 2-D numpy arrays. Each element of a list
    corresponds to a sentence; each row of a 2-D array corresponds to a frame.
    In the case where the pfile doesn't contain labels, "label" will be None.
    """

    with open(filename, "rb") as f:
        # Read header
        # Assuming all data are consistent
        for line in f:
            tokens = line.split()
            if tokens[0] == "-pfile_header":
                headerSize = int(tokens[4])
            elif tokens[0] == "-num_sentences":
                nSentences = int(tokens[1])
            elif tokens[0] == "-num_frames":
                nFrames = int(tokens[1])
            elif tokens[0] == "-first_feature_column":
                cFeature = int(tokens[1])
            elif tokens[0] == "-num_features":
                nFeatures = int(tokens[1])
            elif tokens[0] == "-first_label_column":
                cLabel = int(tokens[1])
            elif tokens[0] == "-num_labels":
                nLabels = int(tokens[1])
            elif tokens[0] == "-format":
                format = tokens[1].replace("d", "i")
            elif tokens[0] == "-end":
                break
        nCols = len(format)
        dataSize = nFrames * nCols

        # Read sentence index
        f.seek(headerSize + dataSize * 4)
        index = struct.unpack(">%di" % (nSentences + 1), f.read(4 * (nSentences + 1)))

        # Read data
        f.seek(headerSize)
        feature = []
        label = []
        sen = 0
        for i in xrange(nFrames):
            if i == index[sen]:
                feature.append([])
                label.append([])
                sen += 1
            data = struct.unpack("<" + format, f.read(4 * nCols))
            feature[-1].append(data[cFeature : cFeature + nFeatures])
            label[-1].append(data[cLabel : cLabel + nLabels])
        feature = [numpy.array(x, dtype=numpy.float32) for x in feature]
        print("shape of feature vector:", numpy.shape(feature[0]))
        label = [numpy.array(x, dtype=numpy.float32) for x in label] if nLabels > 0 else None
        print("shape of label vector:", numpy.shape(label[0]))

    return (feature, label)

def writePfile(filename, feature, label = None):
    """
    Writes "feature" and "label" to a pfile. Both inputs "feature" and "label"
    should be lists of 2-D numpy arrays. Each element of a list corresponds
    to a sentence; each row of a 2-D array corresponds to a frame. In the case
    where there is only one label per frame, the elements of the "label" list
    can be 1-D arrays.
    """

    nSentences = len(feature)
    nFrames = sum(len(x) for x in feature)
    nFeatures = len(numpy.array(feature[0][0]).ravel())
    nLabels = len(numpy.array(label[0][0]).ravel()) if label is not None else 0
    nCols = 2 + nFeatures + nLabels
    headerSize = 32768
    dataSize = nFrames * nCols

    with open((filename), "wb") as f:
        # Write header
        f.write("-pfile_header version 0 size %d\n" % headerSize)
        f.write("-num_sentences %d\n" % nSentences)
        f.write("-num_frames %d\n" % nFrames)
        f.write("-first_feature_column 2\n")
        f.write("-num_features %d\n" % nFeatures)
        f.write("-first_label_column %d\n" % (2 + nFeatures))
        f.write("-num_labels %d\n" % nLabels)
        f.write("-format dd" + "f" * nFeatures + "d" * nLabels + "\n")
        f.write("-data size %d offset 0 ndim 2 nrow %d ncol %d\n" % (dataSize, nFrames, nCols))
        f.write("-sent_table_data size %d offset %d ndim 1\n" % (nSentences + 1, dataSize))
        f.write("-end\n")

        # Write data
        f.seek(headerSize)
        for i in xrange(nSentences):
            #g.write("%d",i)
            for j in xrange(len(feature[i])):
                f.write(struct.pack("<2i", i, j))
                f.write(struct.pack("<%df" % nFeatures, *numpy.array(feature[i][j]).ravel()))
                if label is not None:
                    f.write(struct.pack("<%di" % nLabels, *numpy.array(label[i][j]).ravel()))

        # Write sentence index
        index = numpy.cumsum([0] + [len(x) for x in feature])
        f.write(struct.pack(">%di" % (nSentences + 1), *index))
