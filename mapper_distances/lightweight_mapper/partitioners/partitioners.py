"""
Contains partition objects for building covers.
"""


from __future__ import division
import numpy as np


class PartitionerBase():
    """

    """

    def __init__(self):
        pass

    def fit_transform(self, filter_values, resolution,
                      overlap, description):
        raise NotImplementedError

    def remove_duplicates(self, partition_list):
        return_list = []
        for i, p1 in enumerate(partition_list):
            for j, p2 in enumerate(partition_list[i + 1:]):
                if p1 == p2:
                    break
            else:
                return_list.append(p1)


class UniformPartitioner(PartitionerBase):
    """
    Adapted from functor.cover.IntersectionData.makeUniformIntervals()
    """

    def fit_transform(self, filter_values, resolution,
                      overlap, description):
        """

        Parameters
        ----------
        filter_values
        resolution
        overlap
        description

        Returns
        -------

        """
        mn = filter_values.min()
        mx = filter_values.max()

        if resolution == 1 or mn == mx:
            new_set = set(range(filter_values.shape[0]))
            new_desc = [{'min': mn,
                         'max': mx,
                         'description': description}]
            partition = Partition(new_set, new_desc)
            return Cover([partition])

        DELTA = (mx - mn) / (2 * (resolution - 1))
        bottom = mn - DELTA
        top = mx + DELTA
        ilen = (top - bottom) / (resolution * 1.0)
        delta = (ilen * overlap) / (2.0 * (1.0 - overlap))

        partition_list = []
        for i in range(resolution):
            lowerI = (bottom + (ilen * i)) - delta
            upperI = (bottom + (ilen * (i + 1))) + delta

            vals = np.where((filter_values >= lowerI) & (filter_values <= upperI))[0]
            new_set = set(vals)
            new_desc = [{'min': lowerI,
                         'max': upperI,
                         'description': description}]
            partition = Partition(new_set, new_desc)
            partition_list.append(partition)

        return Cover(partition_list)


class EqualizedPartitioner(PartitionerBase):
    """
    Adapted from functor.cover.IntersectionData.makeConstantCountIntervals()    """

    def fit_transform(self, filter_values, resolution,
                      overlap, description):
        """

        Parameters
        ----------
        filter_values
        resolution
        overlap
        description

        Returns
        -------

        """
        mn = filter_values.min()
        mx = filter_values.max()
        if resolution == 1 or mn == mx:
            new_set = set(range(filter_values.shape[0]))
            new_desc = [{'min': mn,
                         'max': mx,
                         'description': description}]
            partition = Partition(new_set, new_desc)
            return Cover([partition])

        argsort_filter = np.argsort(filter_values)
        uniform = UniformPartitioner()
        partition_list = []
        #         for subset in uniform.fit_transform(sps.rankdata(filter_values, 'dense') - 1, resolution,
        #                                      overlap, description).partitions_:
        for subset in uniform.fit_transform(np.arange(filter_values.shape[0]), resolution,
                                            overlap, description).partitions_:
            new_set = argsort_filter[list(subset.members_)]
            new_desc = [{'min': min(filter_values[new_set]),
                         'max': max(filter_values[new_set]),
                         'description': description}]
            partition = Partition(set(new_set), new_desc)
            partition_list.append(partition)
        return Cover(partition_list)


class RandomPartitioner(PartitionerBase):
    def fit_transform(self, filter_values, resolution, gain,
                      description='Random'):
        pass


class PartitionExpander():
    """
    """

    def __init__(self):
        pass

    def fit_transform(self, cover, metric_space, r):
        """ Expands partitions in cover by radius r around points in partition
        """
        assert r >= 0, 'Expansion distance cannot be negative'
        partitions = []
        for partition in cover.partitions_:
            description = partition.description_ + [{'expanded_by': r}]
            members = set(np.where(metric_space[list(partition.members_), :] <= r)[1])
            partitions.append(Partition(members, description))
        return Cover(partitions)



