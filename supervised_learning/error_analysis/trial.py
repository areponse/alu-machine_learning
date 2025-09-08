def checkPossibility(nums) -> bool:
        counter = 0
        prev = -float('inf')
        for i in range(len(nums)-1):
            if nums[i+1] < nums[i]:
                counter += 1
                if i > 0:
                    prev = nums[i-1]
                if prev > nums[i+1]:
                    print(prev, nums[i+1])
                    counter += 1
            if counter > 1:
                
                return False
        return True


print(checkPossibility([5,7,1,8]))