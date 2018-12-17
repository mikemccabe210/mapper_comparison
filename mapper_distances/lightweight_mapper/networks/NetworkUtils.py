"""
Contains sub-objects necessary to build networks
"""

from __future__ import division
import numpy as np

class Cover():
    """ Cover of discrete metric space. Contains list of partitions.

    """
    def __init__(self, partitions):
        """

        Parameters
        ----------
        partitions
        """
#         self.partitions_ = self.remove_duplicates(partitions)
        self.partitions_ = partitions

    def cross(self, other_cover):
        """

        Parameters
        ----------
        other_cover

        Returns
        -------
        Other
        """
        temp_cover = []
        for other in other_cover.partitions_:
            for partition in self.partitions_:
                temp_cover.append(partition.intersection(other))
        return Cover(temp_cover)

    def remove_duplicates(self, partition_list):
        return_list = []
        for i, p1 in enumerate(partition_list):
            for j, p2 in enumerate(partition_list[i+1:]):
                if p1.members_ and p1.members_ == p2.members_:
                    break
            else:
                return_list.append(p1)
        return return_list


class Partition():
    """ Partition of set

    Contains members and bounds

    """
    def __init__(self, subset, subset_description = []):
        """

        Parameters
        ----------
        subset - set, required
            Set containing the row indices of the
        subset_description - list, optional
            list of dictionaries containing description and boundary conditions for the partition
        """
        self.members_ = subset
        self.description_ = subset_description

    def intersection(self, other_partition):
        """

        Parameters
        ----------
        other_partition

        Returns
        -------

        """
        combined_members = self.members_.intersection(other_partition.members_)
        combined_description = self.description_ + other_partition.description_

        return Partition(combined_members, combined_description)

    def is_empty(self):
        return not self.members_

    def __eq__(self, other):
        self.members_ == other

    def __lt__(self, other):
        self.members_ < other

    def __gt__(self, other):
        self.members_ > other
