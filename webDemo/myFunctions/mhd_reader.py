"""
This is an mhd reader class for reading mhd fies

Partially derived from mhd_utils

Original Licence:
The MIT License (MIT)

CopyrighT (c) 2015 Peter Fischer & authors of previous versions (Bing Jian, Price Jackson)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import os
import sys
import indexed_gzip as igzip


class MHD_reader:
    """Docstring for MHD_reader. """

    def __init__(self, mhd_filename):
        """TODO: to be defined. """

        (self._path, self._mhd_file) = os.path.split(mhd_filename)
        self._mhd_header_dict = self.read_meta_header(mhd_filename)
        self._data_file_name = self._get_data_file_name()
        # open the file for the lifetime of the object
        # if there is a gzip compressed raw-file, use that!
        if os.path.exists(self._data_file_name + ".gz"):
            self._data_file_name = self._data_file_name + ".gz"
            if not os.path.exists(self._data_file_name + "idx"):
                # the gzip index does not exist yet. Build and save it
                print("building index for {}".format(self._data_file_name))
                self._data_file_handle = igzip.IndexedGzipFile(self._data_file_name)
                self._data_file_handle.build_full_index()
                self._data_file_handle.export_index(self._data_file_name + "idx")
            else:
                # The fastest load in the west: Compressed file with prebuilt index
                self._data_file_handle = igzip.IndexedGzipFile(self._data_file_name, index_file=self._data_file_name + "idx")
            self._need_to_read_to_buffer = True
        else:
            # there is no compressed file. Do it the old fashioned way
            self._data_file_handle = open(self._data_file_name, 'rb')
            self._need_to_read_to_buffer = False

        self._number_dimensions = int(self._mhd_header_dict['NDims'])
        self._shape = list(self._mhd_header_dict['DimSize'])
        self._number_frames = self._shape[self._number_dimensions - 1]
        self._number_scanlines = self._shape[1]
        self._data_type = self._get_element_data_type()
        self._number_element_channels = self._parse_number_channels()
        self._element_bytes = np.dtype(self._data_type).itemsize

    def __del__(self):
        # Close file handle at destruction time
        self._data_file_handle.close()

    def _get_data_file_name(self):
        """ Get path to data file
        Returns: path to data file as string

        """

        if self._path:
            data_file_name = os.path.join(
                self._path, self._mhd_header_dict['ElementDataFile'])
        else:
            data_file_name = self._mhd_header_dict['ElementDataFile']

        return data_file_name

    def get_data_shape(self):
        """Return mhd data shape.
        Returns: TODO

        """
        return self._shape

    @property
    def data_type(self):
        return self._data_type

    @property
    def number_frames(self):
        return self._number_frames

    @property
    def number_scanlines(self):
        return self._number_scanlines

    @property
    def number_element_channels(self):
        return self._number_element_channels

    @property
    def mhd_path(self):
        return self._path

    def read_meta_header(self, filename):
        """Read meta data from MHD file
        Args:
            filename (string): The path to the mhd header file
        Returns: a dictionary of meta data from meta header file

        """
        fileIN = open(filename, "r")
        line = fileIN.readline()

        meta_dict = {}
        tag_set = []
        tag_set.extend([
            'ObjectType', 'NDims', 'DimSize', 'ElementType', 'ElementDataFile',
            'ElementNumberOfChannels'
        ])
        tag_set.extend([
            'BinaryData', 'BinaryDataByteOrderMSB', 'CompressedData',
            'CompressedDataSize'
        ])
        tag_set.extend([
            'Offset', 'CenterOfRotation', 'AnatomicalOrientation',
            'ElementSpacing', 'TransformMatrix'
        ])
        tag_set.extend([
            'Comment', 'SeriesDescription', 'AcquisitionDate',
            'AcquisitionTime', 'StudyDate', 'StudyTime'
        ])

        tag_flag = [False] * len(tag_set)
        while line:
            tags = str.split(line, '=')
            for i in range(len(tag_set)):
                tag = tag_set[i]
                if (str.strip(tags[0]) == tag) and (not tag_flag[i]):
                    # print(tags[1])
                    content = str.strip(tags[1])
                    if tag in [
                            'ElementSpacing', 'Offset', 'CenterOfRotation',
                            'TransformMatrix'
                    ]:
                        meta_dict[tag] = [float(s) for s in content.split()]
                    elif tag in ['NDims', 'ElementNumberOfChannels']:
                        meta_dict[tag] = int(content)
                    elif tag in ['DimSize']:
                        meta_dict[tag] = [int(s) for s in content.split()]
                    elif tag in [
                            'BinaryData', 'BinaryDataByteOrderMSB',
                            'CompressedData'
                    ]:
                        if content == "True":
                            meta_dict[tag] = True
                        else:
                            meta_dict[tag] = False
                    else:
                        meta_dict[tag] = content
                    tag_flag[i] = True
            line = fileIN.readline()
        fileIN.close()
        return meta_dict

    def _get_element_data_type(self):
        """get np_type of mhd_data

        Args:
            _check_meta_dict_type (TODO): TODO

        Returns: nd_type

        """
        meta_dict = self._mhd_header_dict

        if meta_dict['ElementType'] == 'MET_FLOAT':
            np_type = np.float32
        elif meta_dict['ElementType'] == 'MET_DOUBLE':
            np_type = np.float64
        elif meta_dict['ElementType'] == 'MET_CHAR':
            np_type = np.byte
        elif meta_dict['ElementType'] == 'MET_UCHAR':
            np_type = np.ubyte
        elif meta_dict['ElementType'] == 'MET_SHORT':
            np_type = np.short
        elif meta_dict['ElementType'] == 'MET_USHORT':
            np_type = np.ushort
        elif meta_dict['ElementType'] == 'MET_INT':
            np_type = np.int32
        elif meta_dict['ElementType'] == 'MET_UINT':
            np_type = np.uint32
        else:
            raise NotImplementedError(
                "ElementType " + meta_dict['ElementType'] + " not understood.")

        return np_type

    def _parse_number_channels(self):
        """TODO: Docstring for _parse_number_channels.
        Returns: TODO

        """
        meta_dict = self._mhd_header_dict
        if "ElementNumberOfChannels" in meta_dict:
            element_channels = int(meta_dict["ElementNumberOfChannels"])
        else:
            element_channels = 1

        return element_channels

    def _read_pixel(self, frame_number, scan_line_number, pixel_number):
        """Get the data from one MHD pixel

        Args:
            location_tuple (TODO): TODO

        Returns: TODO

        """

        scan_line_depth = self._shape[0]

        num_frame_elements = np.prod(self._shape[:self._number_dimensions - 1])
        frame_offset = frame_number * num_frame_elements

        scan_line_offset = scan_line_depth * scan_line_number
        offset = (frame_offset + scan_line_offset + pixel_number
                  ) * self._element_bytes * self._number_element_channels

        shape = (1, self._number_element_channels)

        self._data_file_handle.seek(offset)
        if self._need_to_read_to_buffer:
            data = np.frombuffer(
                self._data_file_handle.read(np.prod(shape) * self._element_bytes),
                count=np.prod(shape),
                dtype=self._data_type)
        else:
            data = np.fromfile(
                self._data_file_handle,
                count=np.prod(shape),
                dtype=self._data_type)
        if data.size != np.prod(shape):
            tb = sys.exc_info()[2]
            raise ValueError('Failure loading image ' + str(frame_number) +
                             ', scanline ' + str(scan_line_number) +
                             ' from "' + self._data_file_name +
                             '" for reading').with_traceback(tb)
        data = data.reshape(shape)

        return data

    def _read_scan_line(self, frame_number, scan_line_number):
        """Get one scan line from MHD

        Args:
            location_tuple (TODO): TODO

        Returns: TODO

        """

        depth = self._shape[0]

        num_frame_elements = np.prod(self._shape[:-1])
        frame_offset = frame_number * num_frame_elements

        scan_line_offset = depth * scan_line_number
        offset = (frame_offset + scan_line_offset
                  ) * self._element_bytes * self._number_element_channels

        #  Always right num_channels <18-07-18, Walter Simson> #
        out_shape = (depth, self._number_element_channels)
        """
        print(offset)
        print("pre-seek: {}".format(self._data_file_handle.tell()))
        print("post-seek: {}".format(self._data_file_handle.tell()))
        print(
            "minus offset: {}".format(self._data_file_handle.tell() - offset))
        """
        if not self._data_file_handle.readable():
            raise OSError("file currently not readable")

        self._data_file_handle.seek(offset)  # Go to start of data from SOF
        if self._need_to_read_to_buffer:
            data = np.frombuffer(
                self._data_file_handle.read(np.prod(out_shape) * self._element_bytes),
                count=np.prod(out_shape),
                dtype=self._data_type)
        else:
            data = np.fromfile(
                self._data_file_handle,
                count=np.prod(out_shape),
                dtype=self._data_type)

        if data.size == 0:
            tb = sys.exc_info()[2]
            raise AttributeError(
                'Datasize is 0: Failure loading frame {} scanline {} from {} for reading\n shape is {}.\n data.size: {}\nCurrent reader pos is {} while file size is {} \n Offset was: {}'.
                format(frame_number, scan_line_number, self._data_file_name,
                       out_shape, data.size, self._data_file_handle.tell(),
                       self._data_file_handle.seek(0, os.SEEK_END),
                       offset)).with_traceback(tb)

        if data.size != np.prod(out_shape):
            tb = sys.exc_info()[2]
            raise AttributeError(
                'Failure loading frame {} scanline {} from {} for reading\n shape is {}.\n data.size: {}\nCurrent reader pos is {} while file size is {} \n Offset was: {}'.
                format(frame_number, scan_line_number, self._data_file_name,
                       out_shape, data.size, self._data_file_handle.tell(),
                       self._data_file_handle.seek(0, os.SEEK_END),
                       offset)).with_traceback(tb)

        data = data.reshape(out_shape)

        return data

    def _read_frame(self, frame_number):
        """Get one frame from MHD file

        Args:
            location_tuple (TODO): TODO

        Returns: TODO

        """

        # cast product to int, to avoid overflows in int32
        num_frame_elements = int(np.prod(self._shape[:-1]))
        frame_offset = frame_number * num_frame_elements
        offset = frame_offset * self._element_bytes * self._number_element_channels

        shape = (*self._shape[:-1][::-1], self._number_element_channels)

        self._data_file_handle.seek(offset)
        if self._need_to_read_to_buffer:
            data = np.frombuffer(
                self._data_file_handle.read(np.prod(shape) * self._element_bytes),
                count=np.prod(shape),
                dtype=self._data_type)
        else:
            data = np.fromfile(
                self._data_file_handle,
                count=np.prod(shape),
                dtype=self._data_type)

        if data.size == 0:
            tb = sys.exc_info()[2]
            raise AttributeError(
                'Datasize is 0 in file {} with offset {}\n The file size is {}'.
                format(self._data_file_name, offset,
                       self._data_file_handle.seek(
                           0, os.SEEK_END))).with_traceback(tb)

        if data.size != np.prod(shape):
            tb = sys.exc_info()[2]
            raise AttributeError(
                'Failure loading frame {} from {} for reading\n shape is {}.\n data.size: {}\nCurrent reader pos is {} while file size is {} \n Offset was: {}'.
                format(frame_number, self._data_file_name,
                       shape, data.size, self._data_file_handle.tell(),
                       self._data_file_handle.seek(0, os.SEEK_END),
                       offset)).with_traceback(tb)

        data = data.reshape(shape)
        return data

    def _read_series(self):
        """Return a full seriese of frames

        Returns: TODO

        """
        #  TODO: fix multiple reads bug <12-03-18, Walter Simson> #
        if self._number_element_channels == 1:
            shape = self._shape[::-1]  # Reverse order
        else:
            shape = (*self._shape[::-1], self._number_element_channels)

        self._data_file_handle.seek(0)  # Reset previous seeks
        if self._need_to_read_to_buffer:
            data = np.frombuffer(
                self._data_file_handle.read(np.prod(self._shape) * self._element_bytes),
                count=np.prod(self._shape),
                dtype=self._data_type)
        else:
            data = np.fromfile(
                self._data_file_handle,
                count=np.prod(self._shape),
                dtype=self._data_type)

        if data.size != np.prod(self._shape):
            tb = sys.exc_info()[2]
            raise AttributeError(
                'Failure loading series from "' + self._data_file_name +
                '" for reading\n self._shape is {} while data.size is {}'.
                format(data.size, np.prod(self._shape))).with_traceback(tb)
        data = data.reshape(shape)
        return data

    def read_chunk(self, location_tuple=()):
        """Read a chunk based on tupple parameters passed to the read function

        Args:
            chunk_tuple (tuple): a tuple of "chunk indexes" of the form
                                 (frame, scan_line, pixel)
        e.g.
        ```
            self.read_chunk(1,)            # first frame tuple
            self.read_chunk(1, 3)          # first frame third scan line tuple
            self.read_chunk(1, 3, 5)       # first frame third scan line 5th pixel
            self.read_chunk(None)          # entire series_data

        ```
        Returns: np_array of data

        """

        #  TODO: do a better job of this <12-03-18, Walter Simson> #
        #  IDEA: use tupple unpacking to this end and overload sigle read function <19-07-18, Walter Simson> #
        # Parse tuple input
        if len(location_tuple) == 3:
            data = self._read_pixel(*location_tuple)
        elif len(location_tuple) == 2:
            data = self._read_scan_line(*location_tuple)
        elif len(location_tuple) == 1:
            data = self._read_frame(*location_tuple)
        elif not location_tuple:
            data = self._read_series()
        else:
            raise ValueError(
                "location tuple recieved does not have an acceptable format")

        return data
